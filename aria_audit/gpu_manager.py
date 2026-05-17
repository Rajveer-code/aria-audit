"""Sequential GPU model lifecycle manager.

The 8 GB VRAM ceiling on RTX 4060 makes co-loading any two of
{Qwen3 8B, BGE-M3, HHEM 2.1, distil-large-v3, ColPali} unsafe.

Contract (per plan v3 mutual-exclusion table):
- Qwen3 8B is the always-resident anchor (~5.6 GB at Q4_K_M + 4K KV).
- At most ONE of the auxiliary models may be co-resident at a time.
- Every load checks `torch.cuda.mem_get_info()` and refuses if free VRAM
  is below `model_size_gb + SAFETY_MARGIN_GB`.
- Loads are serialized via a process-local lock; concurrent callers queue.

This module imports lazily — `torch` is only required when a CUDA path runs.
The dataclass surface works without GPU for unit-testing on CI/CPU.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Generator

logger = logging.getLogger(__name__)

SAFETY_MARGIN_GB: float = 0.5
GB: int = 1024 ** 3


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a managed model."""

    name: str
    est_vram_gb: float
    anchor: bool = False  # if True, always-resident; load once, never unload


# Canonical specs — adjust est_vram_gb after Phase 0 VRAM gate test
QWEN3_8B = ModelSpec("qwen3-8b-q4_K_M", est_vram_gb=5.6, anchor=True)
BGE_M3 = ModelSpec("bge-m3", est_vram_gb=1.1)
HHEM_21 = ModelSpec("hhem-2.1", est_vram_gb=0.9)
DISTIL_WHISPER_V3 = ModelSpec("distil-large-v3", est_vram_gb=1.5)
COLPALI = ModelSpec("colpali", est_vram_gb=1.4)  # offline subprocess only


class VRAMExceeded(RuntimeError):
    """Raised when a load would exceed the safety margin."""


class GPUManager:
    """Process-local sequential loader. Single shared instance via `get_manager()`."""

    _instance: "GPUManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded: dict[str, object] = {}
        self._aux_holder: str | None = None  # which auxiliary model is currently co-resident

    @classmethod
    def get(cls) -> "GPUManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def free_vram_gb() -> float:
        """Return free VRAM in GB, or +inf if no CUDA (allows CPU-only test paths)."""
        try:
            import torch
        except ImportError:
            return float("inf")
        if not torch.cuda.is_available():
            return float("inf")
        free_b, _total_b = torch.cuda.mem_get_info()
        return free_b / GB

    @staticmethod
    def peak_vram_gb() -> float:
        try:
            import torch
        except ImportError:
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / GB

    @staticmethod
    def reset_peak_stats() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _check_budget(self, spec: ModelSpec) -> None:
        free = self.free_vram_gb()
        need = spec.est_vram_gb + SAFETY_MARGIN_GB
        if free < need:
            raise VRAMExceeded(
                f"Refusing to load {spec.name}: need {need:.2f} GB free "
                f"(model {spec.est_vram_gb:.2f} + {SAFETY_MARGIN_GB:.2f} margin), "
                f"have {free:.2f} GB. Currently co-resident aux: {self._aux_holder!r}."
            )

    @contextmanager
    def acquire(
        self,
        spec: ModelSpec,
        loader: Callable[[], object],
        unloader: Callable[[object], None] | None = None,
    ) -> Generator[object, None, None]:
        """Load `spec` (refusing on budget violation), yield the loaded object, unload on exit.

        Anchor models (Qwen3) stay resident across calls — repeated acquire() returns
        the cached instance without reloading. Auxiliary models follow load-use-unload,
        and the manager enforces that only one non-anchor aux is co-resident at a time.
        """
        with self._lock:
            if spec.anchor:
                if spec.name in self._loaded:
                    obj = self._loaded[spec.name]
                else:
                    self._check_budget(spec)
                    obj = loader()
                    self._loaded[spec.name] = obj
                    logger.info("Loaded anchor %s (peak %.2f GB)", spec.name, self.peak_vram_gb())
                try:
                    yield obj
                finally:
                    pass  # anchor never unloads
                return

            # Auxiliary path: enforce mutex
            if self._aux_holder is not None and self._aux_holder != spec.name:
                raise VRAMExceeded(
                    f"Cannot load {spec.name}: aux slot held by {self._aux_holder}. "
                    "Release the current aux before loading another."
                )
            self._check_budget(spec)
            obj = loader()
            self._aux_holder = spec.name
            self._loaded[spec.name] = obj
            logger.info(
                "Loaded aux %s (free now %.2f GB, peak %.2f GB)",
                spec.name,
                self.free_vram_gb(),
                self.peak_vram_gb(),
            )
            try:
                yield obj
            finally:
                try:
                    if unloader is not None:
                        unloader(obj)
                finally:
                    self._loaded.pop(spec.name, None)
                    self._aux_holder = None
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    logger.info("Unloaded aux %s (free %.2f GB)", spec.name, self.free_vram_gb())


def get_manager() -> GPUManager:
    """Singleton accessor."""
    return GPUManager.get()
