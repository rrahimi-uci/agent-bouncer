"""Cover the OpenAI guard predict paths with a fake client (no network)."""

from agent_bouncer.core.schema import Decision
from agent_bouncer.evaluation.openai_guards import OpenAIChatGuard, OpenAIModerationGuard


class _Cats:
    def model_dump(self):
        return {"hate": True, "violence": False}


class _ModResult:
    flagged = True
    categories = _Cats()


class _ModClient:
    class moderations:
        @staticmethod
        def create(**k):
            class R:
                results = [_ModResult()]
            return R()


def test_moderation_predict_flagged():
    g = OpenAIModerationGuard()
    g._client = _ModClient()
    v = g.predict("something hateful")
    assert v.decision == Decision.UNSAFE and v.score == 1.0


def _chat_client(content):
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    class R:
                        choices = [type("C", (), {"message": type("M", (), {"content": content})})]
                    return R()
    return Client()


def test_chat_predict_parses_unsafe():
    g = OpenAIChatGuard("gpt-4o-mini")
    g._client = _chat_client('{"decision":"unsafe","hazard":"hate"}')
    assert g.predict("x").decision == Decision.UNSAFE


def test_chat_predict_unparseable_defaults_safe():
    g = OpenAIChatGuard("gpt-4o-mini")
    g._client = _chat_client("not json at all")
    assert g.predict("x").decision == Decision.SAFE  # never inflate FPR on junk output


def test_chat_predict_content_policy_refusal_is_unsafe():
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    raise RuntimeError("invalid_prompt: limited access to this content")

    g = OpenAIChatGuard("gpt-5.2", reasoning_effort="low")
    g._client = Client()
    assert g.predict("x").decision == Decision.UNSAFE  # provider refusal → unsafe
