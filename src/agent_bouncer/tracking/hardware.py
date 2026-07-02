"""Capture the hardware + runtime environment for a run, so model performance can be
compared **fairly** across machines (a 500 ms decoder on a laptop CPU is not the same
number on an A100).

Fully cross-platform — CPU brand, memory, and accelerator are detected per OS
(Linux · macOS · Windows), preferring ``psutil`` when installed and falling back to
stdlib probes. Every probe is best-effort and never raises; unknown fields return
``None`` rather than crashing.
"""

from __future__ import annotations

import os
import platform
import subprocess

_SYSTEM = platform.system()  # 'Linux' | 'Darwin' | 'Windows' | ...


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - missing tool / timeout / permission
        return ""


def _mem_gb() -> float | None:
    """Total physical memory in GB, per OS."""
    try:  # best, cross-platform
        import psutil

        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass

    if _SYSTEM == "Linux":
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        except (ValueError, OSError, AttributeError):
            pass
        try:  # /proc/meminfo (kB)
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) * 1024 / 1e9, 1)
        except OSError:
            pass
    elif _SYSTEM == "Darwin":
        val = _run(["sysctl", "-n", "hw.memsize"])
        if val.isdigit():
            return round(int(val) / 1e9, 1)
    elif _SYSTEM == "Windows":
        try:  # GlobalMemoryStatusEx via ctypes — no deps, no subprocess
            import ctypes

            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MEM()
            st.dwLength = ctypes.sizeof(_MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            if st.ullTotalPhys:
                return round(st.ullTotalPhys / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    return None


def _cpu_name() -> str:
    """Human CPU brand string, per OS (falls back to arch)."""
    if _SYSTEM == "Linux":
        try:  # /proc/cpuinfo 'model name' has the real brand; platform.processor() is just the arch
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif _SYSTEM == "Darwin":
        val = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if val:
            return val
    elif _SYSTEM == "Windows":
        val = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if val:
            return val
        val = _run(["wmic", "cpu", "get", "name"])  # last resort
        lines = [x.strip() for x in val.splitlines() if x.strip() and "Name" not in x]
        if lines:
            return lines[0]
    return platform.processor() or platform.machine() or "unknown"


def _accelerator() -> dict:
    """GPU/accelerator via torch — CUDA (Linux/Windows) or MPS (macOS)."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "gpu": "none", "gpu_name": None, "gpu_count": 0, "gpu_memory_gb": None}
    gpu, gpu_name, gpu_count, gpu_mem = "cpu", None, 0, None
    try:
        if torch.cuda.is_available():
            gpu = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            try:
                gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:  # noqa: BLE001
                pass
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu, gpu_name, gpu_count = "mps", "Apple Silicon (MPS)", 1
    except Exception:  # noqa: BLE001
        pass
    return {"torch": torch.__version__, "gpu": gpu, "gpu_name": gpu_name,
            "gpu_count": gpu_count, "gpu_memory_gb": gpu_mem}


def hardware_info() -> dict:
    """A JSON-serializable snapshot of the machine + runtime (works on any OS)."""
    return {
        "platform": platform.platform(),
        "system": _SYSTEM,
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_gb(),
        "python": platform.python_version(),
        **_accelerator(),
    }


def _gb(x) -> str:
    """Format a GB float cleanly: 40.0 -> '40 GB', 38.7 -> '38.7 GB'."""
    return f"{x:.1f}".rstrip("0").rstrip(".") + " GB"


def hardware_label(info: dict | None = None) -> str:
    """Compact one-line label, e.g. 'NVIDIA A100 (40 GB) · 32 cores · 128 GB · py3.11'
    or 'Intel Xeon · 8 cores · 32 GB · py3.12' when there is no GPU."""
    info = info or hardware_info()
    gpu = info.get("gpu")
    if gpu in ("cuda", "mps") and info.get("gpu_name"):
        dev = info["gpu_name"]
        if info.get("gpu_memory_gb"):
            dev += f" ({_gb(info['gpu_memory_gb'])})"
    else:
        dev = info.get("cpu") or "CPU"  # no accelerator → show the CPU brand
    mem = _gb(info["memory_gb"]) if info.get("memory_gb") else "?"
    return f"{dev} · {info.get('cpu_count', '?')} cores · {mem} · py{info.get('python', '?')}"
