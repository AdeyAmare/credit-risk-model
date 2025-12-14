import unittest
import pandas as pd
from src.utils.helpers import get_raw_data_path, ensure_dirs, load_raw_data
from src.eda.eda import EDAHelper
from src.config.config import Config

class TestHelpersEDA(unittest.TestCase):

    def test_get_raw_data_path(self):
        """Check that get_raw_data_path returns a Path object."""
        path = get_raw_data_path("data.csv")
        self.assertTrue(isinstance(path, type(Config.DATA_DIR)))

    def test_ensure_dirs(self):
        """Check that ensure_dirs creates the folder if it doesn't exist."""
        ensure_dirs()  # just run it, no exception should occur

    def test_eda_helper(self):
        """Test EDAHelper methods on an example DataFrame."""
        df = pd.DataFrame({
            "num": [1, 2, 3, 4, 1000],
            "cat": ["a", "b", "a", "b", "a"]
        })
        eda = EDAHelper(df)
        results = eda.run_all()

        # Simple checks
        self.assertEqual(results["overview"]["n_rows"], 5)
        self.assertEqual(results["overview"]["n_cols"], 2)
        self.assertIn("num", results["numeric_summary"].index)
        self.assertIn("cat", results["categorical_summary"])
        self.assertIn("Outliers Count", results["outlier_summary"].columns)

if __name__ == "__main__":
    unittest.main()
