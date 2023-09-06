from unittest import TestCase

import numpy as np
import pandas as pd

from utilities.dataGenerationUtilities import create_price_variables

test_dataframe = pd.DataFrame({'Average': [1, 2, 4, 2]})

correct_result = pd.DataFrame({
    'Average': [1, 2, 4, 2],
    '1period_shifted_price': [np.nan, 1.0, 2.0, 4.0],
    '1period_change_in_price': [np.nan, 1.0, 2.0, -2.0],
    '1period_percentage_change_in_price': [np.nan, 100.0, 100.0, -50.0],
    'sum_of_absolute_percentage_price_changes_over_1_periods': [np.nan, 100.0, 100.0, 50.0],
    '2period_shifted_price': [np.nan, np.nan, 1.0, 2.0],
    '2period_change_in_price': [np.nan, np.nan, 3.0, 0.0],
    '2period_percentage_change_in_price': [np.nan, np.nan, 300.0, 0.0],
    'sum_of_absolute_percentage_price_changes_over_2_periods': [np.nan, np.nan, np.nan, 300.0]
})

print()


class TestCreatePriceVariables(TestCase):
    def test_create_price_variables(self):
        result_df = pd.DataFrame(create_price_variables(test_dataframe, range(2)))

        for key in list(correct_result.keys()):
            np.testing.assert_array_equal(result_df[key].values, correct_result[key].values)