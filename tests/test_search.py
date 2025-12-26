import types

import src.search as search_mod


def test_build_search_orchestrator_prefers_instance_override():
    agent = types.SimpleNamespace()
    # put an instance-level builder into __dict__ to simulate monkeypatch
    agent.__dict__["_build_search_orchestrator"] = lambda: "OVERRIDE"
    assert search_mod.build_search_orchestrator(agent) == "OVERRIDE"
