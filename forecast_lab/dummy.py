"""
Dummy models, implementing simplistic forecasting strategies.
"""

import numpy
import scipy

class MeanForecast:
    """Forecasts the mean of the time series"""

    def fit(self, ts, ext_vars=None):
        self.mean = numpy.mean(ts)
        return self

    def forecast(self, steps, ext_vars=None):
        return numpy.full(steps, self.mean)


class LinearForecast:
    """
    Fits a linear function to the given time series and extrapolates it for the forecast.
    """

    def fit(self, ts, ext_vars):
        slope, intercept, _, _, _ = scipy.stats.linregress(numpy.arange(len(ts)), ts)
        self.slope = slope
        self.intercept = intercept
        return self

    def forecast(self, steps, ext_vars=None):
        return self.slope * numpy.arange(steps) + self.intercept
