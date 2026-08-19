import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_trend.features import attach_lagged_fred_features


class MacroPointInTimeTests(unittest.TestCase):
    def test_initial_release_is_available_next_business_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fred_DGS2.csv"
            pd.DataFrame(
                {
                    "series_id": ["DGS2", "DGS2"],
                    "date": ["2026-08-03", "2026-08-04"],
                    "value": [4.1, 4.2],
                    "realtime_start": ["2026-08-03", "2026-08-04"],
                }
            ).to_csv(path, index=False)
            frame = pd.DataFrame(
                {
                    "session_date": [
                        pd.Timestamp("2026-08-03").date(),
                        pd.Timestamp("2026-08-04").date(),
                        pd.Timestamp("2026-08-05").date(),
                    ]
                }
            )
            result = attach_lagged_fred_features(frame, directory)
            self.assertTrue(pd.isna(result.loc[0, "macro_DGS2_level"]))
            self.assertEqual(float(result.loc[1, "macro_DGS2_level"]), 4.1)
            self.assertEqual(float(result.loc[2, "macro_DGS2_level"]), 4.2)


if __name__ == "__main__":
    unittest.main()

