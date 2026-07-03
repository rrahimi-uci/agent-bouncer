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


def test_hf_token_is_mirrored_to_both_aliases(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_mirror\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    load_env(env)
    # set under both names so any HF library picks it up regardless of which it reads
    assert os.environ["HF_TOKEN"] == "hf_mirror"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_mirror"


def test_hf_token_mirror_respects_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "from-real-env")
    load_env(tmp_path / "nope.env")  # no file, but mirror the real env var across
    assert os.environ["HF_TOKEN"] == "from-real-env"
