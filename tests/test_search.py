import types

import src.search as search_mod


def test_build_search_orchestrator_prefers_instance_override():
    agent = types.SimpleNamespace()
    # put an instance-level builder into __dict__ to simulate monkeypatch
    agent.__dict__["_build_search_orchestrator"] = lambda: "OVERRIDE"
    assert search_mod.build_search_orchestrator(agent) == "OVERRIDE"


def test_run_search_rounds_delegates(monkeypatch):
    called = {}

    class FakeOrch:
        def run(self, context, should_search, primary_search_query):
            called["context"] = context
            called["should_search"] = should_search
            called["primary_search_query"] = primary_search_query
            return ["res1"], {"kw"}

    monkeypatch.setattr(search_mod, "build_search_orchestrator", lambda agent: FakeOrch())

    agent = types.SimpleNamespace(chains={}, embedding_client=None)
    ctx = types.SimpleNamespace(
        current_datetime="d",
        current_year="y",
        current_month="m",
        current_day="dd",
        conversation_text="conv",
        prior_responses_text="p",
    )

    res, kws = search_mod.run_search_rounds(agent, ctx, "user", True, "seed", None, None, set())
    assert res == ["res1"]
    assert kws == {"kw"}
    # Verify run received proper arguments
    assert called["should_search"] is True
    assert called["primary_search_query"] == "seed"
    assert called["context"].user_query == "user"
    assert called["context"].current_datetime == "d"
