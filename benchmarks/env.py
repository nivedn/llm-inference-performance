"""Capture the hardware/software environment of a benchmark run.

Every result file embeds this snapshot so any number can be traced back to the
machine, driver, library versions, environment flags and git commit that produced it.

Run standalone to inspect the current machine:
    uv run python -m benchmarks.env
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "triton",
    "vllm",
    "tensorrt_llm",
    "onnxruntime",
    "numpy",
)

# Environment variables that change execution or timing behaviour.
# CUDA_LAUNCH_BLOCKING=1, for example, silently serializes every kernel launch.
TRACKED_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "VLLM_ATTENTION_BACKEND",
)

NVIDIA_SMI_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "clocks.max.sm",
    "clocks.max.memory",
    "power.limit",
    "persistence_mode",
    "pci.bus_id",
)


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_info() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "logical_cores": os.cpu_count()}
    if sys.platform == "darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_bytes"] = int(mem) if mem else None
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                info["model"] = next(
                    (ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")),
                    None,
                )
            with open("/proc/meminfo") as f:
                kb = next((int(ln.split()[1]) for ln in f if ln.startswith("MemTotal")), None)
                info["memory_bytes"] = kb * 1024 if kb else None
        except OSError:
            pass
    return info


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _nvidia_smi() -> list[dict[str, str]] | None:
    out = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    return [
        dict(zip(NVIDIA_SMI_FIELDS, (v.strip() for v in line.split(","))))
        for line in out.splitlines()
    ]


def _torch_info() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None

    info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "name": p.name,
                    "compute_capability": f"{p.major}.{p.minor}",
                    "total_memory_bytes": p.total_memory,
                    "multiprocessor_count": p.multi_processor_count,
                }
            )
    return info


def capture_environment() -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "os": {"system": platform.system(), "release": platform.release(), "platform": platform.platform()},
        "python": sys.version.split()[0],
        "cpu": _cpu_info(),
        "git": _git_info(),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi(),
        "env_vars": {k: os.environ.get(k) for k in TRACKED_ENV_VARS},
    }


if __name__ == "__main__":
    print(json.dumps(capture_environment(), indent=2))
