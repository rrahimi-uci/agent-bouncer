import agent_bouncer.hardware as HW
from agent_bouncer.hardware import hardware_info, hardware_label
from agent_bouncer.models_registry import BASE_MODELS, catalog, get_base_model
from agent_bouncer.training_runner import build_config


def test_new_models_registered_with_rl():
    for key in ("deepseek-r1-1.5b", "smollm2-1.7b", "gemma-1b"):
        m = get_base_model(key)
        assert m.arch == "decoder"
        assert "sft" in m.techniques and "grpo" in m.techniques  # same FT + RL as other SLMs
    assert BASE_MODELS["gemma-1b"].gated is True


def test_catalog_shape():
    cat = catalog()
    keys = {c["key"] for c in cat}
    assert {"distilbert", "qwen3-0.6b", "deepseek-r1-1.5b", "smollm2-1.7b", "gemma-1b"} <= keys
    assert all({"hf_id", "arch", "params", "techniques", "gated"} <= set(c) for c in cat)


def test_build_config_encoder_vs_decoder():
    enc = build_config("distilbert", "sft", "train.jsonl", "out", {"epochs": 3}, 42)
    assert enc["arch"] == "encoder" and enc["train"]["epochs"] == 3
    dec = build_config("smollm2-1.7b", "sft", "train.jsonl", "out", {"max_steps": 20}, 42)
    assert dec["arch"] == "decoder" and dec["train"]["max_steps"] == 20
    assert "bf16" not in dec["train"]
    grpo = build_config("deepseek-r1-1.5b", "grpo", "train.jsonl", "out", {"max_steps": 40}, 42)
    assert grpo["mode"] == "reasoning" and grpo["grpo"]["steps"] == 40


def test_hardware_info_has_expected_keys():
    info = hardware_info()
    for k in ("platform", "system", "machine", "cpu", "cpu_count", "memory_gb",
              "python", "gpu", "gpu_count", "gpu_memory_gb"):
        assert k in info
    assert isinstance(hardware_label(info), str) and "cores" in hardware_label(info)


def test_hardware_probes_work_on_every_os(monkeypatch):
    # the CPU/memory probes must not raise regardless of the host OS
    for sysname in ("Linux", "Darwin", "Windows"):
        monkeypatch.setattr(HW, "_SYSTEM", sysname)
        assert isinstance(HW._cpu_name(), str) and HW._cpu_name()
        mem = HW._mem_gb()
        assert mem is None or mem > 0


def test_label_shows_cpu_when_no_accelerator():
    info = {"gpu": "cpu", "gpu_name": None, "cpu": "Intel Xeon", "cpu_count": 8,
            "memory_gb": 32.0, "python": "3.12"}
    label = hardware_label(info)
    assert "Intel Xeon" in label and "8 cores" in label and "32 GB" in label


def test_label_shows_gpu_with_memory():
    info = {"gpu": "cuda", "gpu_name": "NVIDIA A100", "gpu_memory_gb": 40.0,
            "cpu": "x", "cpu_count": 32, "memory_gb": 128.0, "python": "3.11"}
    assert "NVIDIA A100 (40 GB)" in hardware_label(info)
