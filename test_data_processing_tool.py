import unittest
import pandas as pd
from io import StringIO
from data_processing_tool import read_csv, clean_data, transform_data, compute_statistics, plot_data  # Replace 'your_module' with the name of your file

class TestDataProcessingTool(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        data = """A,B,C
                    1,2,3
                    4,,6
                    7,8,100
                    10,11,12
                    13,14,15
                    16,17,18
                    19,20,21"""
        self.df = pd.read_csv(StringIO(data))

    def test_read_csv(self):
        df = read_csv('data.csv')  # Ensure this file exists for the test
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertNotIn(float('nan'), cleaned_df['B'].values)  # Ensure there are no NaN values
        self.assertLessEqual(cleaned_df['C'].max(), 21)  # Ensure outliers are removed

    def test_transform_data_normalize(self):
        normalized_df = transform_data(self.df[['A', 'B', 'C']], 'normalize')
        self.assertEqual(normalized_df['A'].min(), 0)
        self.assertEqual(normalized_df['A'].max(), 1)

    def test_transform_data_standardize(self):
        standardized_df = transform_data(self.df[['A', 'B', 'C']], 'standardize')
        self.assertAlmostEqual(standardized_df['A'].mean(), 0, places=1)  # Mean should be close to 0

    def test_compute_statistics(self):
        stats = compute_statistics(self.df[['A', 'B', 'C']].dropna())
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('mode', stats)

    def test_plot_data(self):
        # This test would not normally be automated as it generates plots.
        # Here, we can just check if it runs without error.
        try:
            plot_data(self.df[['A', 'B', 'C']].dropna(), 'histogram')
        except Exception as e:
            self.fail(f"plot_data raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
