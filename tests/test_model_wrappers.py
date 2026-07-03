"""Cover EncoderGuard/DecoderGuard load+predict and the decoder training-asset loader.

Uses REAL torch with FAKE Hugging Face model/tokenizer objects injected via sys.modules,
so the tensor math runs for real while no weights are downloaded.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _fake_transformers(decode_text='{"decision": "safe", "hazard": "none"}'):
    m = types.ModuleType("transformers")

    class FakeTok:
        pad_token = None
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, path, **k):
            return cls()

        def __call__(self, text, **k):
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

        def decode(self, ids, skip_special_tokens=True):
            return decode_text

    class FakeSeq:
        config = types.SimpleNamespace(id2label={0: "safe", 1: "hate"})

        @classmethod
        def from_pretrained(cls, path, **k):
            return cls()

        def eval(self):
            return self

        def __call__(self, **inputs):
            return types.SimpleNamespace(logits=torch.tensor([[0.2, 3.0]]))  # → unsafe (hate)

    class FakeCausal:
        def __init__(self):
            self.config = types.SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, path, **k):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, **k):
            return torch.tensor([[1, 2, 3, 4, 5, 6]])

    m.AutoTokenizer = FakeTok
    m.AutoModelForSequenceClassification = FakeSeq
    m.AutoModelForCausalLM = FakeCausal
    return m


def test_encoder_guard_loads_and_predicts(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    from agent_bouncer.core.schema import Decision
    from agent_bouncer.models.encoder import EncoderGuard
    v = EncoderGuard("path", name="enc").predict("hello")
    assert v.model == "enc" and v.decision == Decision.UNSAFE  # softmax([.2,3])→class1
    assert v.latency_ms is not None and 0.0 <= v.score <= 1.0


def test_decoder_guard_safe_verdict(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    from agent_bouncer.core.schema import Decision
    from agent_bouncer.models.decoder import DecoderGuard
    v = DecoderGuard("path", name="dec", device="cpu").predict("hello")
    assert v.model == "dec" and v.decision == Decision.SAFE


def test_decoder_guard_fails_closed_on_junk(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(decode_text="not json"))
    from agent_bouncer.core.schema import Decision
    from agent_bouncer.models.decoder import DecoderGuard
    v = DecoderGuard("path", name="dec", mode="reasoning", device="cpu").predict("hello")
    assert v.decision == Decision.UNSAFE and v.score == 0.5  # unparseable → fail closed


def test_decoder_resolves_device_when_unset(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    from agent_bouncer.models.decoder import DecoderGuard
    g = DecoderGuard("path")  # device=None → auto-resolve (mps if available else cpu)
    g.predict("hi")
    assert g.device in ("mps", "cpu")


def test_load_decoder_training_assets(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    from agent_bouncer.training.runtime import load_decoder_training_assets, training_device
    model, tok, device, bf16 = load_decoder_training_assets("some/model", train_cfg={"bf16": False})
    assert tok.pad_token == "</s>"  # set from eos when missing
    assert model.config.use_cache is False and device in ("mps", "cpu", "cuda")
    assert training_device() in ("mps", "cpu", "cuda")
