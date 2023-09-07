from unittest import TestCase

import numpy as np
import pandas as pd

from utilities.dataGenerationUtilities import create_volume_variables

test_dataframe = pd.DataFrame({'Volume': [1, 2, 4, 2, 4, 3, 2, 1, 1, 1]})

correct_result = {
    'Volume': [1, 2, 4, 2, 4, 3, 2, 1, 1, 1],
    '1period_shifted_volume': [np.nan, 1.0, 2.0, 4.0, 2.0, 4.0, 3.0, 2.0, 1.0, 1.0],
    '1period_change_in_volume': [np.nan, 1.0, 2.0, -2.0, 2.0, -1.0, -1.0, -1.0, 0.0, 0.0],
    '1period_percentage_change_in_volume': [np.nan, 100.0, 100.0, -50.0, 100.0, -25.0, -33.333333, -50.0, 0.0, 0.0],
    'sum_of_absolute_percentage_volume_changes_over_last_1_periods': [np.nan, 100.0, 100.0, 50.0, 100.0, 25.0, 33.33333, 50.0,
                                                                0.0, 0.0],
    '2period_shifted_volume': [np.nan, np.nan, 1.0, 2.0, 4.0, 2.0, 4.0, 3.0, 2.0, 1.0],
    '2period_change_in_volume': [np.nan, np.nan, 3.0, 0.0, 0.0, 1.0, -2.0, -2.0, -1.0, 0.0],
    '2period_percentage_change_in_volume': [np.nan, np.nan, 300.0, 0.0, 0.0, 50.0, -50.0, -66.66667, -50.0, 0.0],
    'sum_of_absolute_percentage_volume_changes_over_last_2_periods': [np.nan, np.nan, 200.0, 150.0, 150.0,
                                                                125.0, 58.33333, 83.33333, 50.0, 0.0]
}

correct_result = pd.DataFrame(correct_result)


class TestCreatevolumeVariables(TestCase):
    def test_create_volume_variables(self):
        result_df = pd.DataFrame(create_volume_variables(test_dataframe, range(1, 3)))

        for key in list(correct_result.keys()):
            np.testing.assert_allclose(result_df[key].values, correct_result[key].values, atol=1e-6)
