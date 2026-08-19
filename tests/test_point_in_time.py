import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_trend.cognition import Thesis, attach_thesis_features, assert_point_in_time_safe


class PointInTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timeline = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-08-05T13:00:00Z",
                        "2026-08-05T14:00:00Z",
                        "2026-08-05T15:00:00Z",
                    ],
                    utc=True,
                )
            }
        )

    def test_thesis_only_activates_after_creation(self) -> None:
        thesis = Thesis.from_dict(
            {
                "thesis_id": "live",
                "created_at": "2026-08-05T14:00:00Z",
                "valid_from": "2026-08-05T14:00:00Z",
                "expires_at": "2026-08-05T16:00:00Z",
                "target_symbol": "AAPL",
                "regime_code": "semis_to_target_rotation",
                "expected_direction": 1,
                "prior_confidence": 70,
                "is_retrospective": False,
            }
        )
        result = attach_thesis_features(self.timeline, [thesis], "AAPL")
        self.assertEqual(result.loc[0, "cognition_active"], 0.0)
        self.assertEqual(result.loc[1, "cognition_active"], 1.0)

    def test_retrospective_case_is_never_a_training_feature(self) -> None:
        thesis = Thesis.from_dict(
            {
                "thesis_id": "retro",
                "created_at": "2026-08-06T00:00:00Z",
                "valid_from": "2026-07-01T00:00:00Z",
                "expires_at": "2026-07-31T00:00:00Z",
                "target_symbol": "AAPL",
                "regime_code": "semis_to_target_rotation",
                "expected_direction": 1,
                "prior_confidence": 90,
                "is_retrospective": True,
            }
        )
        result = attach_thesis_features(self.timeline, [thesis], "AAPL")
        self.assertEqual(float(result["cognition_active"].sum()), 0.0)

    def test_first_seen_cannot_precede_event(self) -> None:
        events = pd.DataFrame(
            {
                "event_time": pd.to_datetime(["2026-08-05T15:00:00Z"], utc=True),
                "first_seen_time": pd.to_datetime(["2026-08-05T14:59:00Z"], utc=True),
            }
        )
        with self.assertRaises(ValueError):
            assert_point_in_time_safe(events, [])


if __name__ == "__main__":
    unittest.main()

