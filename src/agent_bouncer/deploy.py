"""Deployment (roadmap phase 6): quantize/export the trained guard and measure
on-device latency — the whole point of a *small* guard.

The command builders are pure (unit-tested); the runners shell out to external
toolchains (llama.cpp, mlx_lm, optimum). `measure_latency` works on any `Guard`.
"""

from __future__ import annotations

import subprocess

from .guard import Guard
from .metrics import _percentile


def build_gguf_command(
    model_dir: str,
    out_file: str,
    quant: str = "Q4_K_M",
    llama_cpp_dir: str = "llama.cpp",
) -> list[str]:
    """llama.cpp HF->GGUF conversion command (run after cloning llama.cpp)."""
    return [
        "python",
        f"{llama_cpp_dir}/convert_hf_to_gguf.py",
        model_dir,
        "--outfile",
        out_file,
        "--outtype",
        quant.lower(),
    ]


def build_mlx_convert_command(model_dir: str, out_dir: str, quantize: bool = True) -> list[str]:
    """mlx_lm conversion command (Apple Silicon)."""
    cmd = ["python", "-m", "mlx_lm.convert", "--hf-path", model_dir, "--mlx-path", out_dir]
    if quantize:
        cmd.append("-q")
    return cmd


def run_command(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=True).returncode


def export_onnx(model_dir: str, out_dir: str) -> str:
    """Export an encoder guard to ONNX (fast CPU/browser inference). Requires optimum."""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pip install optimum[onnxruntime]") from exc
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    model.save_pretrained(out_dir)
    return out_dir


def measure_latency(guard: Guard, samples: list[str], warmup: int = 5) -> dict[str, float]:
    """Return per-request latency stats for a guard (the on-device story)."""
    for s in samples[:warmup]:
        guard.predict(s)
    latencies = [guard.predict(s).latency_ms or 0.0 for s in samples]
    return {
        "n": float(len(latencies)),
        "p50_ms": _percentile(latencies, 0.5),
        "p95_ms": _percentile(latencies, 0.95),
        "mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
