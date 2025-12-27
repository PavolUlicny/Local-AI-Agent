from __future__ import annotations

from src.agent import Agent
from src.config import AgentConfig


def test_reduce_context_and_rebuild_does_not_increase() -> None:
    # set a small context to ensure we don't grow it when reducing
    cfg = AgentConfig(assistant_num_ctx=1500, robot_num_ctx=1500, assistant_num_predict=800)
    agent = Agent(cfg)
    orig_ctx = min(cfg.assistant_num_ctx, cfg.robot_num_ctx)
    orig_predict = cfg.assistant_num_predict
    # call reduce; should not increase context (and should decrease to >= 1024)
    agent._reduce_context_and_rebuild("planning", "planning")
    new_ctx = min(cfg.assistant_num_ctx, cfg.robot_num_ctx)
    new_predict = cfg.assistant_num_predict
    assert new_ctx <= orig_ctx
    assert new_ctx >= 1024
    assert agent.rebuild_counts["planning"] == 1
    # predict should not have grown above original predictable max
    assert new_predict <= orig_predict


def test_char_budget_behavior() -> None:
    # test mapping tokens->chars behavior for various contexts
    cfg = AgentConfig(assistant_num_ctx=512, robot_num_ctx=512)
    agent = Agent(cfg)
    # With small ctx_tokens, the budget candidate is small but min is 1024
    # 512 * 4 * 0.8 = 1638, so budget = min(10000, max(1024, 1638)) = 1638
    assert agent._char_budget(10000) == 1638

    cfg = AgentConfig(assistant_num_ctx=1500, robot_num_ctx=1500)
    agent = Agent(cfg)
    # ctx_tokens ~1500 -> int(1500*4*0.8)=4800, so budget = min(base, 4800)
    assert agent._char_budget(10000) == 4800
    assert agent._char_budget(100) == 100


def test_context_similarity_edge_cases() -> None:
    cfg = AgentConfig()
    agent = Agent(cfg)
    # None embeddings yield zero
    assert agent._context_similarity(None, None, None) == 0.0
    # when both question and topic embeddings are present, the max similarity is returned
    # candidate aligns with question but not topic -> max should be 1.0
    assert agent._context_similarity([1.0, 0.0], [1.0, 0.0], [0.0, 1.0]) == 1.0
    # candidate aligns with topic more than question -> pick the topic similarity
    assert agent._context_similarity([0.0, 1.0], [0.0, 0.2], [0.0, 1.0]) == 1.0
