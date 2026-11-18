import numpy as np
import pytest

from qnavi.errors import ConfigError
from qnavi.table import QTable


def test_initial_values_are_filled():
    table = QTable(3, 2, initial_value=0.5)
    assert np.all(table.values == 0.5)
    assert table.visits.sum() == 0


def test_update_moves_towards_target():
    table = QTable(2, 2)
    delta = table.update(0, 1, target=2.0, alpha=0.25)
    assert table[0, 1] == pytest.approx(0.5)
    assert delta == pytest.approx(2.0)
    assert table.visits[0, 1] == 1


def test_best_actions_returns_all_ties():
    table = QTable(1, 3)
    table.values[0] = [1.0, 3.0, 3.0]
    assert table.best_actions(0).tolist() == [1, 2]


def test_coverage_counts_visited_pairs():
    table = QTable(2, 2)
    table.update(0, 0, 1.0, 0.1)
    assert table.coverage() == pytest.approx(0.25)


def test_copy_is_independent():
    table = QTable(2, 2)
    clone = table.copy()
    clone.update(1, 1, 5.0, 1.0)
    assert table[1, 1] == 0.0


def test_load_arrays_checks_shape():
    table = QTable(2, 2)
    with pytest.raises(ConfigError):
        table.load_arrays(np.zeros((3, 2)), np.zeros((3, 2), dtype=int))


def test_greedy_policy_and_state_values():
    table = QTable(2, 3)
    table.values[:] = [[0, 2, 1], [5, 0, 0]]
    assert table.greedy_policy().tolist() == [1, 0]
    assert table.state_values().tolist() == [2.0, 5.0]


def test_invalid_dimensions_rejected():
    with pytest.raises(ConfigError):
        QTable(0, 2)
