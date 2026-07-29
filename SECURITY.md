# Security Policy

This repository processes only trusted local research data. Persistent files
include integrity checks for accidental corruption, but they are not
authenticated against a malicious source.

## Checkpoint files

Checkpoint files (`checkpoint_merged.pkl`) use Python's `pickle` format
and should only be loaded from trusted local sources.  Never load a
checkpoint file received from an untrusted party — pickle deserialisation
can execute arbitrary code.

## Sigma analysis database

`sigma_pool.sqlite3` is a persistent cache of derived σ-analysis results.
Rows are validated with a payload checksum, an arithmetic identity check,
prime tests, and (for partial records) a prime-pool prefix certificate before
reuse.  Invalid or incompatible rows fall back to an ordinary cache miss.

The database is not a checkpoint and is not required for mathematical
correctness.  It may be deleted to force a cold analysis, but the program must
first be stopped.  Delete the main database and any matching `-wal` and `-shm`
files together; deleting SQLite files while the process is running can lose
pending cache writes or damage the cache.

## Persistent plan cache

`plan_cache/` contains derived filtered-prime arrays and hierarchical
superblock products. Entries use compatibility keys, size bounds, SHA-256
checksums, locked staging directories, `fsync`, and atomic publication.
Incomplete or accidentally corrupt entries are ignored and rebuilt.

Like checkpoints, this directory is trusted local input rather than an
authenticated format. Do not copy plan entries from an untrusted source:
SHA-256 detects accidental changes but does not prove who created a file.
The cache may be deleted to recover disk space while the program is stopped;
it is not needed to resume the search and does not contain authoritative
mathematical results.

## Supported versions

Only the latest commit on the `main` branch is actively maintained.
Operational deletion and recovery procedures are documented in
[`docs/OUTPUTS_AND_RECOVERY.md`](docs/OUTPUTS_AND_RECOVERY.md).
