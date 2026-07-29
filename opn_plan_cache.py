"""Validated persistent storage for sigma-pool block plans.

Filtered prime arrays are exposed as read-only NumPy memory maps.  Superblock
products are always deserialised into ``mpz`` objects before a plan is used, so
the GCD hot path never performs file I/O or integer decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid
from typing import Sequence

from gmpy2 import mpz
import numpy as np


PLAN_CACHE_SCHEMA_VERSION = 3
PLAN_CACHE_SEMANTICS_VERSION = 3
_MANIFEST_FILE = "manifest.json"
_PRIMES_FILE = "primes.bin"
_PRODUCTS_FILE = "products.bin"
_IO_CHUNK_BYTES = 8 * 1024 * 1024


class PlanCacheError(RuntimeError):
    """Base class for persistent-plan cache failures."""


class PlanCacheValidationError(PlanCacheError):
    """A cache entry is incomplete, incompatible, or corrupt."""


class PlanCacheBusyError(PlanCacheError):
    """Another process is currently constructing the same plan."""


@dataclass(frozen=True, slots=True)
class PlanCacheKey:
    """Everything that determines one logical hierarchical plan."""

    pool_digest: str
    prime_limit: int
    source_start: int
    source_count: int
    filter_kind: str
    filter_order: int
    dtype: str
    block_size: int
    superblock_fanout: int

    def canonical_payload(self) -> bytes:
        return json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")

    @property
    def slug(self) -> str:
        return hashlib.sha256(self.canonical_payload()).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedPlan:
    """A validated plan payload ready for ``PrimeBlockPlan`` construction."""

    eligible_primes: np.ndarray | np.memmap | None
    superblock_products: tuple[mpz, ...]
    prime_count: int
    leaf_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_IO_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows' CRT rejects fsync() on a read-only descriptor.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _close_memmap(mapping: np.memmap | None) -> None:
    if mapping is None:
        return
    mmap_object = getattr(mapping, "_mmap", None)
    if mmap_object is not None:
        mmap_object.close()


class _PlanFileLock:
    """Small cross-platform advisory lock retained for the cache lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            handle.close()
            raise PlanCacheBusyError(
                f"plan cache entry is busy: {self.path.name}"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


class PlanCacheBuild:
    """One locked, invisible plan construction transaction."""

    def __init__(
        self,
        cache: "PersistentPlanCache",
        key: PlanCacheKey,
        lock: _PlanFileLock,
        staging_dir: Path,
    ) -> None:
        self.cache = cache
        self.key = key
        self.lock = lock
        self.staging_dir = staging_dir
        self._finished = False

    @property
    def primes_path(self) -> Path:
        return self.staging_dir / _PRIMES_FILE

    def allocate_primes(
        self,
        prime_count: int,
    ) -> np.ndarray | np.memmap:
        if self.key.filter_order == 2:
            raise PlanCacheError(
                "unfiltered plans do not own a prime-array file"
            )
        if prime_count < 0:
            raise PlanCacheError("prime count must be non-negative")
        dtype = np.dtype(self.key.dtype)
        required = prime_count * dtype.itemsize
        free = shutil.disk_usage(self.cache.root).free
        if free < required + self.cache.minimum_free_bytes:
            raise PlanCacheError(
                "insufficient free space for filtered plan array"
            )
        if prime_count == 0:
            self.primes_path.touch()
            return np.empty(0, dtype=dtype)
        return np.memmap(
            self.primes_path,
            dtype=dtype,
            mode="w+",
            shape=(prime_count,),
        )

    def open_staging_primes(
        self,
        prime_count: int,
    ) -> np.ndarray | np.memmap:
        if prime_count == 0:
            return np.empty(0, dtype=np.dtype(self.key.dtype))
        return np.memmap(
            self.primes_path,
            dtype=np.dtype(self.key.dtype),
            mode="r",
            shape=(prime_count,),
        )

    def commit(
        self,
        *,
        prime_count: int,
        leaf_count: int,
        superblock_products: Sequence[mpz],
        first_prime: int,
        last_prime: int,
    ) -> np.ndarray | np.memmap | None:
        if self._finished:
            raise PlanCacheError("plan build is already finished")
        dtype = np.dtype(self.key.dtype)
        expected_superblocks = (
            leaf_count + self.key.superblock_fanout - 1
        ) // self.key.superblock_fanout
        if len(superblock_products) != expected_superblocks:
            raise PlanCacheError(
                "superblock count does not match the logical plan"
            )

        primes_size = 0
        primes_sha256 = None
        if self.key.filter_order != 2:
            primes_size = prime_count * dtype.itemsize
            if (
                not self.primes_path.is_file()
                or self.primes_path.stat().st_size != primes_size
            ):
                raise PlanCacheError(
                    "filtered prime array has the wrong size"
                )
            _fsync_file(self.primes_path)
            primes_sha256 = _sha256_file(self.primes_path)

        products_path = self.staging_dir / _PRODUCTS_FILE
        products_digest = hashlib.sha256()
        with products_path.open("wb") as handle:
            for raw_product in superblock_products:
                product = int(raw_product)
                if product <= 0:
                    raise PlanCacheError(
                        "superblock products must be positive"
                    )
                payload = product.to_bytes(
                    max(1, (product.bit_length() + 7) // 8),
                    "big",
                )
                header = struct.pack(">I", len(payload))
                handle.write(header)
                handle.write(payload)
                products_digest.update(header)
                products_digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())

        products_size = products_path.stat().st_size
        manifest = {
            "schema_version": PLAN_CACHE_SCHEMA_VERSION,
            "semantics_version": PLAN_CACHE_SEMANTICS_VERSION,
            "key": asdict(self.key),
            "prime_count": int(prime_count),
            "leaf_count": int(leaf_count),
            "superblock_count": len(superblock_products),
            "first_prime": int(first_prime),
            "last_prime": int(last_prime),
            "primes_size": int(primes_size),
            "primes_sha256": primes_sha256,
            "products_size": int(products_size),
            "products_sha256": products_digest.hexdigest(),
        }
        manifest_path = self.staging_dir / _MANIFEST_FILE
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        final_dir = self.cache.entry_path(self.key)
        quarantine = None
        if final_dir.exists():
            quarantine = final_dir.with_name(
                f"{final_dir.name}.invalid-{uuid.uuid4().hex}"
            )
            os.replace(final_dir, quarantine)
        os.replace(self.staging_dir, final_dir)
        self._finished = True
        if quarantine is not None:
            shutil.rmtree(quarantine, ignore_errors=True)
        self.lock.release()

        if self.key.filter_order == 2:
            return None
        if prime_count == 0:
            return np.empty(0, dtype=dtype)
        return np.memmap(
            final_dir / _PRIMES_FILE,
            dtype=dtype,
            mode="r",
            shape=(prime_count,),
        )

    def abort(self) -> None:
        if self._finished:
            return
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        self._finished = True
        self.lock.release()

    def __enter__(self) -> "PlanCacheBuild":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._finished:
            self.abort()
        return False


class PersistentPlanCache:
    """Directory-backed validated plan cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        minimum_free_bytes: int = 2 * 1024**3,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.minimum_free_bytes = int(minimum_free_bytes)
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum free bytes must be non-negative")

    def entry_path(self, key: PlanCacheKey) -> Path:
        return self.root / key.slug

    def begin(self, key: PlanCacheKey) -> PlanCacheBuild:
        lock = _PlanFileLock(self.root / f"{key.slug}.lock")
        lock.acquire()
        try:
            for stale in self.root.glob(f"{key.slug}.tmp-*"):
                if stale.is_dir() and stale.parent == self.root:
                    shutil.rmtree(stale, ignore_errors=True)
            staging_dir = self.root / (
                f"{key.slug}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            )
            staging_dir.mkdir()
        except BaseException:
            lock.release()
            raise
        return PlanCacheBuild(
            self,
            key,
            lock,
            staging_dir,
        )

    def load(self, key: PlanCacheKey) -> LoadedPlan | None:
        entry = self.entry_path(key)
        manifest_path = entry / _MANIFEST_FILE
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self._validate_manifest(key, entry, manifest)
            products = self._load_products(entry, manifest)
            mapping = self._load_primes(entry, manifest)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
            struct.error,
        ) as exc:
            raise PlanCacheValidationError(
                f"invalid plan cache entry {key.slug}: {exc}"
            ) from exc

        return LoadedPlan(
            eligible_primes=mapping,
            superblock_products=products,
            prime_count=int(manifest["prime_count"]),
            leaf_count=int(manifest["leaf_count"]),
        )

    def _validate_manifest(
        self,
        key: PlanCacheKey,
        entry: Path,
        manifest: dict,
    ) -> None:
        if key.filter_kind != "component":
            raise ValueError("unsupported plan filter kind")
        if key.filter_order < 2:
            raise ValueError("invalid cyclotomic component order")
        if manifest["schema_version"] != PLAN_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported plan cache schema")
        if (
            manifest["semantics_version"]
            != PLAN_CACHE_SEMANTICS_VERSION
        ):
            raise ValueError("unsupported plan cache semantics")
        if manifest["key"] != asdict(key):
            raise ValueError("plan cache key mismatch")

        prime_count = int(manifest["prime_count"])
        leaf_count = int(manifest["leaf_count"])
        superblock_count = int(manifest["superblock_count"])
        if prime_count < 0 or leaf_count < 0:
            raise ValueError("plan cache contains negative counts")
        if prime_count > key.source_count:
            raise ValueError("plan cache prime count exceeds its source")
        first_prime = int(manifest["first_prime"])
        last_prime = int(manifest["last_prime"])
        if prime_count == 0:
            if first_prime != 0 or last_prime != 0:
                raise ValueError("empty plan has non-empty boundaries")
        elif (
            first_prime < 3
            or first_prime > last_prime
            or last_prime > key.prime_limit
        ):
            raise ValueError("plan cache prime boundaries are invalid")
        expected_leaf_count = (
            prime_count + key.block_size - 1
        ) // key.block_size
        if leaf_count != expected_leaf_count:
            raise ValueError("plan cache leaf count mismatch")
        expected_superblocks = (
            leaf_count + key.superblock_fanout - 1
        ) // key.superblock_fanout
        if superblock_count != expected_superblocks:
            raise ValueError("plan cache superblock count mismatch")

        products_path = entry / _PRODUCTS_FILE
        max_product_bytes = (
            key.block_size
            * key.superblock_fanout
            * max(1, key.prime_limit.bit_length())
            + 7
        ) // 8
        products_size = int(manifest["products_size"])
        if (
            not products_path.is_file()
            or products_path.stat().st_size
            != products_size
        ):
            raise ValueError("plan cache products size mismatch")
        if products_size > (
            superblock_count * (4 + max_product_bytes)
        ):
            raise ValueError("plan cache products file is oversized")

        if key.filter_order == 2:
            if (
                int(manifest["primes_size"]) != 0
                or manifest["primes_sha256"] is not None
            ):
                raise ValueError(
                    "unfiltered plan unexpectedly owns a prime file"
                )
        else:
            primes_path = entry / _PRIMES_FILE
            expected_size = (
                prime_count * np.dtype(key.dtype).itemsize
            )
            if (
                int(manifest["primes_size"]) != expected_size
                or not primes_path.is_file()
                or primes_path.stat().st_size != expected_size
            ):
                raise ValueError("plan cache prime-array size mismatch")

    def _load_products(
        self,
        entry: Path,
        manifest: dict,
    ) -> tuple[mpz, ...]:
        path = entry / _PRODUCTS_FILE
        digest = hashlib.sha256()
        products = []
        key = manifest["key"]
        max_product_bytes = (
            int(key["block_size"])
            * int(key["superblock_fanout"])
            * max(1, int(key["prime_limit"]).bit_length())
            + 7
        ) // 8
        with path.open("rb") as handle:
            for _ in range(int(manifest["superblock_count"])):
                header = handle.read(4)
                if len(header) != 4:
                    raise ValueError("truncated superblock header")
                digest.update(header)
                (length,) = struct.unpack(">I", header)
                if length < 1 or length > max_product_bytes:
                    raise ValueError(
                        "invalid superblock product length"
                    )
                payload = handle.read(length)
                if len(payload) != length:
                    raise ValueError("truncated superblock product")
                digest.update(payload)
                product = mpz(int.from_bytes(payload, "big"))
                if product <= 0:
                    raise ValueError("non-positive superblock product")
                products.append(product)
            if handle.read(1):
                raise ValueError("trailing bytes in products file")
        if digest.hexdigest() != manifest["products_sha256"]:
            raise ValueError("superblock products checksum mismatch")
        return tuple(products)

    def _load_primes(
        self,
        entry: Path,
        manifest: dict,
    ) -> np.ndarray | np.memmap | None:
        key = manifest["key"]
        if int(key["filter_order"]) == 2:
            return None
        path = entry / _PRIMES_FILE
        if _sha256_file(path) != manifest["primes_sha256"]:
            raise ValueError("filtered prime-array checksum mismatch")
        prime_count = int(manifest["prime_count"])
        if prime_count == 0:
            return np.empty(0, dtype=np.dtype(key["dtype"]))
        mapping = np.memmap(
            path,
            dtype=np.dtype(key["dtype"]),
            mode="r",
            shape=(prime_count,),
        )
        if (
            int(mapping[0]) != int(manifest["first_prime"])
            or int(mapping[-1]) != int(manifest["last_prime"])
        ):
            _close_memmap(mapping)
            raise ValueError("filtered prime-array boundary mismatch")
        return mapping
