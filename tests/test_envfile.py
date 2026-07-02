import os

from agent_bouncer.config.envfile import load_env


def test_load_env_sets_missing_and_respects_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nHF_TOKEN=hf_abc123\nexport OPENAI_API_KEY=\"sk-xyz\"\nBLANK=\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")  # real env must win

    used = load_env(env)
    assert used == str(env.resolve())
    assert os.environ["HF_TOKEN"] == "hf_abc123"          # loaded (quotes/export stripped)
    assert os.environ["OPENAI_API_KEY"] == "already-set"  # setdefault: existing wins


def test_load_env_missing_file_is_noop(tmp_path):
    assert load_env(tmp_path / "nope.env") is None
