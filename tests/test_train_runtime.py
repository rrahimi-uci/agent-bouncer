import pytest

torch = pytest.importorskip("torch")  # torch is an optional (train/eval) extra, absent in CI

from agent_bouncer.train.runtime import decoder_bf16_enabled, decoder_model_load_kwargs  # noqa: E402


def test_decoder_training_defaults_to_bf16_on_mps():
    assert decoder_bf16_enabled({}, "mps") is True
    kwargs = decoder_model_load_kwargs({}, device="mps")
    assert kwargs["dtype"] == torch.bfloat16
    assert kwargs["device_map"] is None
    assert kwargs["low_cpu_mem_usage"] is False


def test_decoder_training_respects_explicit_bf16_false():
    assert decoder_bf16_enabled({"bf16": False}, "mps") is False
    kwargs = decoder_model_load_kwargs({"bf16": False}, device="mps")
    assert kwargs["dtype"] == torch.float32


def test_decoder_training_keeps_fp32_off_mps_by_default():
    assert decoder_bf16_enabled({}, "cpu") is False
    kwargs = decoder_model_load_kwargs({}, device="cpu", trust_remote_code=True)
    assert kwargs["dtype"] == torch.float32
    assert kwargs["trust_remote_code"] is True
