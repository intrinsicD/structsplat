from benchmarks.stage_search import FACTORIAL_DEFAULTS, INFLUENCE_DEFAULTS
from structsplat.config import DEFAULT_INIT_STRATEGY, GenConfig, InitConfig


def test_init_default_is_quadtree_wse_with_measured_offset():
    cfg = InitConfig()

    assert DEFAULT_INIT_STRATEGY == "quadtree_wse"
    assert cfg.strategy == "quadtree_wse"
    assert cfg.flank_offset_frac == 0.0


def test_flanking_keeps_historical_offset_when_selected():
    cfg = InitConfig(strategy="aniso_flanking")
    explicit = InitConfig(strategy="quadtree_wse", flank_offset_frac=0.25)

    assert cfg.flank_offset_frac == 0.5
    assert explicit.flank_offset_frac == 0.25


def test_generation_and_stage_search_follow_shipped_init_default():
    assert GenConfig().init_strategy == "quadtree_wse"
    assert FACTORIAL_DEFAULTS["strategies"][0] == "quadtree_wse"
    assert INFLUENCE_DEFAULTS["strategies"][0] == "quadtree_wse"
    assert "aniso_flanking" in INFLUENCE_DEFAULTS["strategies"]
