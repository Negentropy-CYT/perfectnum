"""
opn_runtime — background performance sampler (RSS, CPU, rate CSV).

Uses a daemon thread to read psutil process metrics on a fixed interval.
The thread never accesses the search heap or mathematical counters.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import psutil


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Lightweight progress state sampled by the background thread."""
    phase: str = "startup"
    states_started: int = 0
    states_completed: int = 0
    frontier_size: int = 0


class RuntimeSampler:
    """Periodic RSS/CPU/rate CSV logger (daemon thread, no search coupling)."""

    def __init__(
        self,
        csv_path: Path,
        *,
        elapsed_offset: float = 0.0,
        interval_seconds: float = 2.0,
        append: bool = False,
    ) -> None:
        self.csv_path = csv_path
        self.elapsed_offset = elapsed_offset
        self.interval_seconds = interval_seconds

        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = ProgressSnapshot()
        self._thread: threading.Thread | None = None

        self._started = 0.0
        self._previous_elapsed = elapsed_offset
        self._previous_completed = 0
        self.sampled_peak_rss = 0

        mode = "a" if append else "w"
        self._file = csv_path.open(
            mode,
            newline="",
            encoding="utf-8",
            buffering=1,
        )
        self._writer = csv.writer(self._file)

        self._wall_start = time.monotonic()

        if not append or csv_path.stat().st_size == 0:
            self._writer.writerow([
                "elapsed_s",
                "wall_s",
                "phase",
                "states_started",
                "states_completed",
                "frontier_size",
                "average_rate",
                "recent_rate",
                "rss_bytes",
                "sampled_peak_rss_bytes",
                "vms_bytes",
                "cpu_percent",
                "thread_count",
                "system_available_bytes",
            ])

    # ── public API ──────────────────────────────────────────

    def start(self) -> None:
        """Start the background sampling daemon thread."""
        self._started = time.monotonic()
        self._process.cpu_percent(None)  # prime the first reading
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-sampler",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        """Set the current run phase label (thread-safe).

        Resets the rate computation baseline so the first sample in the
        new phase does not blend wall-time with the previous phase.
        """
        with self._lock:
            self._snapshot = replace(self._snapshot, phase=phase)
            self._previous_elapsed = (
                self.elapsed_offset
                + time.monotonic()
                - self._started
            )
            self._previous_completed = self._snapshot.states_completed

    def update_progress(
        self,
        *,
        states_started: int,
        states_completed: int,
        frontier_size: int,
    ) -> None:
        """Update progress counters (thread-safe, called from search loop)."""
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                states_started=states_started,
                states_completed=states_completed,
                frontier_size=frontier_size,
            )

    def stop(self) -> None:
        """Signal the thread, take a final sample, join."""
        if not self._stop.is_set():
            self.sample()
            self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self._file.flush()
        self._file.close()

    def sample(self) -> None:
        """Take one sample (called from thread *or* synchronously)."""
        with self._lock:
            snapshot = self._snapshot

        elapsed = (
            self.elapsed_offset
            + time.monotonic()
            - self._started
        )
        elapsed = max(elapsed, self._previous_elapsed)

        memory = self._process.memory_info()
        rss = int(memory.rss)
        vms = int(memory.vms)
        self.sampled_peak_rss = max(self.sampled_peak_rss, rss)

        delta_t = elapsed - self._previous_elapsed
        delta_states = (
            snapshot.states_completed
            - self._previous_completed
        )

        recent_rate = delta_states / delta_t if delta_t > 0 else 0.0
        average_rate = (
            snapshot.states_completed / elapsed
            if elapsed > 0 else 0.0
        )

        self._writer.writerow([
            f"{elapsed:.3f}",
            f"{time.monotonic() - self._wall_start:.3f}",
            snapshot.phase,
            snapshot.states_started,
            snapshot.states_completed,
            snapshot.frontier_size,
            f"{average_rate:.3f}",
            f"{recent_rate:.3f}",
            rss,
            self.sampled_peak_rss,
            vms,
            f"{self._process.cpu_percent(None):.1f}",
            self._process.num_threads(),
            psutil.virtual_memory().available,
        ])

        self._previous_elapsed = elapsed
        self._previous_completed = snapshot.states_completed

    def capture_memory_phase(
        self,
        phases: dict[str, dict[str, int]],
        phase: str,
    ) -> None:
        """Snapshot memory under *phase* in the *phases* dict."""
        phases[phase] = self.capture_memory()

    def capture_memory(self) -> dict[str, int]:
        """Return a labeled memory snapshot for phase tracking."""
        memory = self._process.memory_info()
        rss = int(memory.rss)
        self.sampled_peak_rss = max(self.sampled_peak_rss, rss)

        return {
            "rss_bytes": rss,
            "vms_bytes": int(memory.vms),
            "sampled_peak_rss_bytes": self.sampled_peak_rss,
            "system_available_bytes": int(
                psutil.virtual_memory().available
            ),
        }

    # ── internal ────────────────────────────────────────────

    def _run(self) -> None:
        """Background loop — wake, sample, sleep."""
        while not self._stop.wait(self.interval_seconds):
            self.sample()
