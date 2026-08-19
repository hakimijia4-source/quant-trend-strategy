import unittest

import numpy as np
import pandas as pd

from quant_trend.config import load_config
from quant_trend.environment import _returns_to_go, _trajectory_rewards
from quant_trend.labels import add_forward_labels


class LabelAndRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("config/demo.toml")

    def test_smooth_uptrend_receives_positive_label(self) -> None:
        length = 30
        close = np.linspace(100.0, 103.0, length)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05T14:30:00Z", periods=length, freq="5min"),
                "session_date": ["2026-01-05"] * length,
                "close": close,
                "evidence_score": [80.0] * length,
                "event_active": [1.0] * length,
            }
        )
        labeled = add_forward_labels(frame, self.config)
        self.assertEqual(int(labeled.loc[0, "trend_label"]), 1)
        self.assertGreaterEqual(float(labeled.loc[0, "sample_weight"]), 0.70)

    def test_reward_charges_turnover(self) -> None:
        close = np.asarray([100.0, 101.0, 101.0])
        actions = np.asarray([1, 0, 0], dtype=int)
        rewards = _trajectory_rewards(close, actions, 2.5, 0.0, 0.0)
        gross = np.log(101.0 / 100.0)
        self.assertAlmostEqual(rewards[0], gross - 0.00025, places=8)
        self.assertAlmostEqual(rewards[1], -0.00025, places=8)

    def test_return_to_go_is_backward_sum(self) -> None:
        rewards = np.asarray([1.0, 2.0, 3.0])
        result = _returns_to_go(rewards, 1.0)
        np.testing.assert_allclose(result, [6.0, 5.0, 3.0])


if __name__ == "__main__":
    unittest.main()

