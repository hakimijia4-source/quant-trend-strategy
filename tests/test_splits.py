import unittest

import pandas as pd

from quant_trend.rl_data import walk_forward_splits


class SplitTests(unittest.TestCase):
    def test_dates_do_not_overlap(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=30).date
        frame = pd.DataFrame({"session_date": dates})
        split = walk_forward_splits(
            frame, train_days=15, validation_days=5, test_days=5, step_days=5
        )[0]
        self.assertFalse(set(split.train_dates) & set(split.validation_dates))
        self.assertFalse(set(split.validation_dates) & set(split.test_dates))
        self.assertLess(max(split.train_dates), min(split.validation_dates))
        self.assertLess(max(split.validation_dates), min(split.test_dates))


if __name__ == "__main__":
    unittest.main()

