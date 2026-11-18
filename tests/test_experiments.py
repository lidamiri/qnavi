from qnavi.config import TrainingConfig
from qnavi.experiments import compare_agents, sweep_learning_rate


def test_compare_writes_table_for_selected_agents(quick_config, run_root):
    rows, table = compare_agents(quick_config, run_root, agents=["q_learning", "sarsa"])
    assert {row["agent"] for row in rows} == {"q_learning", "sarsa"}
    assert (run_root / "comparison.csv").exists()
    assert "final_return_mean_last_100" in table


def test_sweep_returns_best_alpha(quick_config, run_root):
    rows, table, best = sweep_learning_rate(quick_config, run_root, alphas=(0.1, 0.5))
    assert [row["alpha"] for row in rows] == [0.1, 0.5]
    assert best in (0.1, 0.5)
    assert (run_root / "sweep.csv").exists()


def test_compare_uses_configured_episode_budget(quick_config, run_root):
    config = quick_config.with_overrides(training=TrainingConfig(episodes=5, max_steps=20))
    rows, _ = compare_agents(config, run_root, agents=["expected_sarsa"])
    assert rows[0]["episodes"] == 5
