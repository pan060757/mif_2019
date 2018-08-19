# this is a dataset from R
import numpy as np
from pyramid.arima import auto_arima
from bokeh.plotting import figure, show, output_notebook
import pandas as pd
# init bokeh
output_notebook()
def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
               yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A',
               forecast_start=None, **kwargs):
    # make truth and forecasts into pandas series
    n_truth = truth.shape[0]
    n_forecasts = forecasts.shape[0]

    # always plot truth the same
    truth = pd.Series(truth, index=np.arange(truth.shape[0]))

    # if no defined forecast start, start at the end
    if forecast_start is None:
        idx = np.arange(n_truth, n_truth + n_forecasts)
    else:
        idx = np.arange(forecast_start, n_forecasts)
    forecasts = pd.Series(forecasts, index=idx)

    # set up the plot
    p = figure(title=title, plot_height=400, **kwargs)
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = yaxis_label

    # add the lines
    p.line(truth.index, truth.values, color=c1, legend='Observed')
    p.line(forecasts.index, forecasts.values, color=c2, legend='Forecasted')
    return p

wineind = np.array([
    # Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
    15136, 16733, 20016, 17708, 18019, 19227, 22893, 23739, 21133, 22591, 26786, 29740,
    15028, 17977, 20008, 21354, 19498, 22125, 25817, 28779, 20960, 22254, 27392, 29945,
    16933, 17892, 20533, 23569, 22417, 22084, 26580, 27454, 24081, 23451, 28991, 31386,
    16896, 20045, 23471, 21747, 25621, 23859, 25500, 30998, 24475, 23145, 29701, 34365,
    17556, 22077, 25702, 22214, 26886, 23191, 27831, 35406, 23195, 25110, 30009, 36242,
    18450, 21845, 26488, 22394, 28057, 25451, 24872, 33424, 24052, 28449, 33533, 37351,
    19969, 21701, 26249, 24493, 24603, 26485, 30723, 34569, 26689, 26157, 32064, 38870,
    21337, 19419, 23166, 28286, 24570, 24001, 33151, 24878, 26804, 28967, 33311, 40226,
    20504, 23060, 23562, 27562, 23940, 24584, 34303, 25517, 23494, 29095, 32903, 34379,
    16991, 21109, 23740, 25552, 21752, 20294, 29009, 25500, 24166, 26960, 31222, 38641,
    14672, 17543, 25453, 32683, 22449, 22316, 27595, 25451, 25421, 25288, 32568, 35110,
    16052, 22146, 21198, 19543, 22084, 23816, 29961, 26773, 26635, 26972, 30207, 38687,
    16974, 21697, 24179, 23757, 25013, 24019, 30345, 24488, 25156, 25650, 30923, 37240,
    17466, 19463, 24352, 26805, 25236, 24735, 29356, 31234, 22724, 28496, 32857, 37198,
    13652, 22784, 23565, 26323, 23779, 27549, 29660, 23356]
).astype(np.float64)

# fitting a stepwise model:
stepwise_fit = auto_arima(wineind, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                          start_P=0, seasonal=True, d=1, D=1, trace=True,
                          error_action='ignore',  # don't want to know if an order does not work
                          suppress_warnings=True,  # don't want convergence warnings
                          stepwise=True)  # set to stepwise
print(stepwise_fit.summary())

in_sample_preds = stepwise_fit.predict_in_sample()
print(in_sample_preds[:10])

show(plot_arima(wineind, in_sample_preds,
                title="Original Series & In-sample Predictions",
                c2='#FF0000', forecast_start=0))