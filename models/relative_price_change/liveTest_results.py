import pandas as pd

from models.relative_price_change.relative_price_change import SD_correct_direction

for ticker in ["XOM", "NVDA", "AMD"]:
    for model in ['liveTests/liveTest_relative_price_change_linear_model_',
                  'liveTests/liveTest_relative_price_change_random_forest_model_',
                  'liveTests/liveTest_relative_price_change_mlp_model_']:

        model_results = pd.read_csv(f'{model}{ticker}_5mins_12M_{ticker}.csv')

        model_results['Correct_Direction'] = model_results.apply(
            lambda x: 1 if x['ActualPriceChange'] * x['PredictedPriceChange'] >= 0 else 0, axis=1)
        model_results['Above_2SD_Correct_Direction'] = model_results.apply(
            lambda x: SD_correct_direction(x['ActualPriceChange'], x['PredictedPriceChange'], x['PriceAboveUpperBB2SD']), axis=1)
        model_results['Above_1SD_Correct_Direction'] = model_results.apply(
            lambda x: SD_correct_direction(x['ActualPriceChange'], x['PredictedPriceChange'], x['PriceAboveUpperBB1SD']), axis=1)
        model_results['Below_2SD_Correct_Direction'] = model_results.apply(
            lambda x: SD_correct_direction(x['ActualPriceChange'], x['PredictedPriceChange'], x['PriceBelowLowerBB2SD']), axis=1)
        model_results['Below_1SD_Correct_Direction'] = model_results.apply(
            lambda x: SD_correct_direction(x['ActualPriceChange'], x['PredictedPriceChange'], x['PriceBelowLowerBB1SD']), axis=1)

        above_two_sd_series = model_results['Above_2SD_Correct_Direction'].dropna()
        above_one_sd_series = model_results['Above_1SD_Correct_Direction'].dropna()
        below_two_sd_series = model_results['Below_2SD_Correct_Direction'].dropna()
        below_one_sd_series = model_results['Below_1SD_Correct_Direction'].dropna()

        print("\nModel: ", f'{model}{ticker}_5mins_12M_{ticker}.csv')
        print(f"Overall Correct_Direction: {model_results['Correct_Direction'].sum() / len(model_results)}")
        print(f"Above_2SD_Correct_Direction: {above_two_sd_series.sum() / len(above_two_sd_series)}")
        print(f"Above_1SD_Correct_Direction: {above_one_sd_series.sum() / len(above_one_sd_series)}")
        print(f"Below_2SD_Correct_Direction: {below_two_sd_series.sum() / len(below_two_sd_series)}")
        print(f"Below_1SD_Correct_Direction: {below_one_sd_series.sum() / len(below_one_sd_series)}\n")
