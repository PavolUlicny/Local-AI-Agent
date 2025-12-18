from __future__ import annotations

from src.agent import Agent
from src.config import AgentConfig
from src import text_utils as T


def test_reduce_context_and_rebuild_does_not_increase() -> None:
    # set a small context to ensure we don't grow it when reducing
    cfg = AgentConfig(assistant_num_ctx=1500, robot_num_ctx=1500, assistant_num_predict=800)
    agent = Agent(cfg)
    orig_ctx = min(cfg.assistant_num_ctx, cfg.robot_num_ctx)
    orig_predict = cfg.assistant_num_predict
    # call reduce; should not increase context (and should decrease to >= 1024)
    agent._reduce_context_and_rebuild("seed", "seed")
    new_ctx = min(cfg.assistant_num_ctx, cfg.robot_num_ctx)
    new_predict = cfg.assistant_num_predict
    assert new_ctx <= orig_ctx
    assert new_ctx >= 1024
    assert agent.rebuild_counts["seed"] == 1
    # predict should not have grown above original predictable max
    assert new_predict <= orig_predict


def test_char_budget_behavior() -> None:
    # test mapping tokens->chars behavior for various contexts
    cfg = AgentConfig(assistant_num_ctx=100, robot_num_ctx=100)
    agent = Agent(cfg)
    # With very small ctx_tokens, the budget candidate is small but min is 1024
    assert agent._char_budget(10000) == min(10000, 1024)

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


def test_pick_seed_query_more_edge_cases() -> None:
    fallback = "original"
    # candidates that are punctuation-only or too short should be skipped
    seed_text = "- * !!\nNo\nQ: a\nSEED:   "
    assert T.pick_seed_query(seed_text, fallback) == fallback

    # uppercase prefix handling
    seed_text = "SEED: Detailed findings about XYZ"
    assert T.pick_seed_query(seed_text, fallback) == "Detailed findings about XYZ"
