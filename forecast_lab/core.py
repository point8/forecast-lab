import random
import logging
import seaborn
import matplotlib.pyplot as plt
import pandas
import math
import numpy
import fbprophet
import scipy
import typing
import sklearn
import itertools
import pandas


def sliding_window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = list(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + [elem,]
        yield result

def transform_to_labelled_points(ts, window_size):
    """
    Given a series of values and a window size, it turns
    them into labelled points X,y by sliding the window over the series.
    """
    feature_columns = [f"x_{i}" for i in range(window_size)]
    columns = feature_columns + ["y"]
    rows = []
    for window in sliding_window(ts, window_size + 1):
        label = window[-1]
        features = window[:-1]
        row = dict(zip(columns, features + [label]))
        rows.append(row)
    data = pandas.DataFrame(rows, columns=columns)
    return data[feature_columns], data["y"]



class ForecastWrapper:
    """
    Provides a common interface for time series forecasting
    algorithms from different libraries - implement a subclass to
    support a new interface.
    """

    def __init__(
        self,
        estimator_class,
        estimator_params={},
        fit_params={},
    ):
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.fit_params = fit_params

    def fit(
        self,
        ts: pandas.Series,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Fit the model to a time series
        """
        return self

    def forecast(
        self,
        steps: int,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Forecast the time series for the given
        number of steps after the end of the
        training series.
        """
        return numpy.zeros(steps)

class StatsmodelsWrapper(ForecastWrapper):
    """Wrapper for forecasting algorithms from statsmodels"""

    def __init__(
        self,
        estimator_class,
        estimator_params={},
        fit_params={},
    ):
        super(StatsmodelsWrapper, self).__init__(
            estimator_class,
            estimator_params,
            fit_params
        )

    def fit(
        self,
        ts: pandas.Series,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Fit the model to a time series
        """
        if ext_vars is not None:
            raise ValueError("external variables are not supported")
        self.model = self.estimator_class(
            ts,
            **self.estimator_params
        ).fit(**self.fit_params)
        return self

    def forecast(
        self,
        steps: int,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Forecast the time series for the given
        number of steps after the end of the
        training series.
        """
        return self.model.forecast(steps)[0]

class ProphetWrapper(ForecastWrapper):
    """
    Wrapper for the fbprophet forecasting.
    """

    def __init__(
        self,
        freq,
        estimator_params={},
        fit_params={},
    ):
        self.freq = freq
        super(ProphetWrapper, self).__init__(
            fbprophet.Prophet,
            estimator_params,
            fit_params
        )

    def fit(
        self,
        ts: pandas.Series,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Fit the model to a time series
        """
        if ext_vars is not None:
            raise ValueError("TODO: pass external variables to Prophet")
        #if not hasattr(ts, "freq"):
        #    raise ValueError("The time series frequency attribute `freq` must be set")
        #self.freq = ts.freq
        # Prophet time series format
        if ext_vars:
            raise NotImplementedError("Adding external variables is not yet implemented")
        ts_prophet = pandas.DataFrame(ts).reset_index()
        ts_prophet.columns = ["ds", "y"]
        self.model = self.estimator_class(
            **self.estimator_params
        ).fit(
            ts_prophet,
            **self.fit_params
        )
        return self

    def forecast(
        self,
        steps: int,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Forecast the time series for the given
        number of steps after the end of the
        training series.
        """
        if ext_vars:
            raise NotImplementedError("Adding external variables is not yet implemented")
        forecast_df = self.model.predict(
            self.model.make_future_dataframe(periods=steps, freq=self.freq)
        )
        forecast_df = forecast_df.set_index("ds")
        ts_forecast = forecast_df[-steps:]["yhat"]
        return ts_forecast

class ScikitLearnWrapper(ForecastWrapper):
    """
    1. Transforms time series into a scikit-learn-compatible supervised learning
      problem and fits a scikit-learn estimator
    2. applies the model recursively to forecast
    """

    def __init__(
        self,
        estimator_class,
        sliding_window_size,
        estimator_params={},
        fit_params={},
    ):
        super(ScikitLearnWrapper, self).__init__(
            estimator_class,
            estimator_params,
            fit_params
        )
        self.sliding_window_size = sliding_window_size

    def fit(
        self,
        ts: pandas.Series,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Fit the model to a time series
        """
        X, y =  transform_to_labelled_points(
            ts,
            window_size=self.sliding_window_size
        )
        if ext_vars is not None:
            ext_X = ext_vars.iloc[self.sliding_window_size:].reset_index(drop=True) # align with labelled points
            X = pandas.concat([X, ext_X], axis=1, ignore_index=True) # concatenate columns
        self.model = self.estimator_class(
            **self.estimator_params
        ).fit(
            X,
            y,
            **self.fit_params
        )
        self.ts_train = ts # save for recursive forecast
        return self

    def forecast(
        self,
        steps: int,
        ext_vars: pandas.DataFrame = None
    ):
        """
        Forecast the time series for the given
        number of steps after the end of the
        training series.
        """
        w = self.sliding_window_size
        ts_tail = list(self.ts_train[-(w):].values)
        ts_forecast = []
        for i in range(steps):
            if ext_vars is not None:
                n_features = w + ext_vars.shape[1] # number of feature columns
                X = numpy.array(ts_tail + list(ext_vars.iloc[i].values)).reshape((1, n_features))
            else:
                X = numpy.array(ts_tail).reshape((1, w))
            y = self.model.predict(
                X
            )
            ts_forecast += [y[0]]
            ts_tail = ts_tail[1:] + [y[0]]

        return numpy.array(ts_forecast)

class RNNWrapper(ForecastWrapper):
    """
    """

    def __init__(
        self,
        estimator_class,
        sliding_window_size,
        epochs,
        estimator_params={},
        fit_params={},

    ):
        super(RNNWrapper, self).__init__(
            estimator_class,
            estimator_params,
            fit_params
        )
        self.sliding_window_size = sliding_window_size
        self.epochs = epochs

    def fit(
        self,
        ts: pandas.Series,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Fit the model to a time series
        """
        if ext_vars is not None:
            raise NotImplementedError("TODO")
        X, y =  transform_to_labelled_points(
            ts,
            window_size=self.sliding_window_size
        )
        n = X.shape[0]  # number of samples
        w = self.sliding_window_size  # window size
        # convert to numpy array
        X = X.values.reshape(n, 1, w)
        y = y.values

        self.network = self.estimator_class(
            **self.estimator_params
        )
        self.network.compile(
            loss="mean_absolute_percentage_error",
            metrics=["mse"],
            optimizer="adam"
        )
        for i in range(self.epochs):
            self.network.fit(
                X,
                y,
                batch_size=1,
                shuffle=False,
                **self.fit_params
            )
            self.network.reset_states()
        self.ts_train = ts # save for recursive forecast
        return self

    def forecast(
        self,
        steps: int,
        ext_vars: pandas.DataFrame = None,
    ):
        """
        Forecast the time series for the given
        number of steps after the end of the
        training series.
        """
        if ext_vars is not None:
            raise NotImplementedError("TODO")
        w = self.sliding_window_size
        ts_tail = list(self.ts_train[-(w):].values)
        ts_forecast = []
        for i in range(steps):
            X_tail = numpy.array(ts_tail).reshape((1, 1, w))
            y = self.network.predict(
                X_tail
            )
            y_hat = y[0, 0]
            ts_forecast += [y_hat]
            ts_tail = ts_tail[1:] + [y_hat]

        return numpy.array(ts_forecast)




class ForecastEvaluation:
    """
    This class implements a performance evaluation for a time series forecasting model.
    Somewhat analogous to cross-validation, it randomly splits the time series k times into
    a training and test series and computes the error of the forecast.
    """

    def __init__(
        self,
        ts: pandas.Series,
        forecasting,
        ext_vars: pandas.DataFrame = None,
        ts_test: pandas.Series = None,
        test_window_size: int = None,
        train_window_size: int = None,
        metrics = None
    ):
        self.forecasting = forecasting
        self.ts = ts
        if self.ts.isnull().any():
            raise ValueError("The time series contains missing values")
        self.ext_vars = ext_vars
        self.ts_test = ts_test
        self.test_window_size = test_window_size
        self.train_window_size = train_window_size
        self.metrics = metrics
        self.metrics_results = []
        self.forecasts = {} # stores forecasts

    def _ts_train_test_split(self):
        """
        Perform a random split of the time series into
            ts_left: the left remainder
            ts_train: the segment used for training
            ts_test: the segment used for testing
            ts_right: the right remainder
        """
        l = max(0, self.train_window_size)
        u = self.ts.shape[0] - self.test_window_size
        split_index = random.randint(l, u)
        ts_train = self.ts[split_index - self.train_window_size : split_index]
        ts_test = self.ts[split_index : split_index+self.test_window_size]
        ts_left = self.ts[:split_index - self.train_window_size]
        ts_right = self.ts[split_index + self.test_window_size:]
        #ts_train.freq = self.ts.freq
        #ts_test.freq = self.ts.freq
        #ts_left.freq = self.ts.freq
        #ts_right.freq = self.ts.freq
        return ts_left, ts_train, ts_test, ts_right

    def _compute_metrics(self, ts_test, ts_forecast):

        score_dict = {}
        for (metric_name, metric) in self.metrics.items():
            score = metric(ts_test, ts_forecast)
            score_dict[metric_name] = score

        self.metrics_results.append(score_dict)


    def _show_plots(self, iteration, ts_left=None, ts_train=None, ts_test=None, ts_right=None, ts_forecast=None):
        # colors for rest, training segment, true segment and forecasted segment
        with seaborn.color_palette("colorblind", 4):
            plt.figure()
            plt.suptitle(f"i={iteration}")
            if ts_left is not None:
                plt.plot(ts_left)
            plt.plot(ts_train)
            plt.plot(ts_test)
            plt.plot(pandas.Series(ts_forecast, index=ts_test.index))
            if ts_right is not None:
                plt.plot(ts_right)


    def _plot_residuals(self, ts_forecast, ts_test):
        fig, ax = plt.subplots()
        x = numpy.arange(len(ts_test))
        residuals = ts_forecast - ts_test
        y_max = max(residuals)
        ax.scatter(x, residuals, s=4)
        ax.vlines(x, numpy.zeros(len(residuals)), residuals)
        ax.set_ylim([-y_max,y_max])
        plt.show()

    def _plot_pulls(self, ts_forecast, ts_test):
        fig, ax = plt.subplots()

        n_sample = len(ts_test)
        n_bins = int(n_sample/5)
        min_bin, max_bin = -5, 5
        bins = numpy.linspace(min_bin, max_bin, n_bins)
        width = (max_bin - min_bin) / (n_bins - 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        residuals = ts_forecast - ts_test
        pulls = residuals / numpy.std(ts_test)

        ax.hist(pulls, bins=bins)
        ax.set_xlim([min_bin,max_bin])

        mean, scale = scipy.stats.norm.fit(data=pulls)
        ax.text(0.01, 0.9, f'µ={mean:.3f}, σ={scale:.3f}', transform=ax.transAxes)

        y_model = scipy.stats.norm.pdf(bin_centers, loc=mean, scale=scale) * n_sample * width
        ax.plot(bin_centers, y_model)
        plt.show()

    def evaluate(
        self,
        k,
        plot_segments=False,
        plot_residuals=False,
        plot_pulls=False
    ):
        """
        Train and evaluate the model on a k-fold split of the time series.
        """

        for iteration in range(k):
            if self.ts_test is None:
                # no test time series has been passed - perform random split
                print(f"======== iteration {iteration} ==========")
                ts_left, ts_train, ts_test, ts_right = self._ts_train_test_split()
            else:
                # test time series has been passed
                ts_test = self.ts_test
                ts_train = self.ts
                ts_left = None
                if self.test_window_size:
                    # use a segment of the given ts
                    raise NotImplementedError("TODO")
                else:
                    ts_right = None
            if self.test_window_size:
                h = self.test_window_size
            else:
                h = len(self.ts_test)
            if self.ext_vars is not None:
                ext_train = self.ext_vars[ts_train.index.min():ts_train.index.max()]
                ext_future = self.ext_vars[ts_train.index.max():]
            else:
                ext_train = None
                ext_future = None
            ts_forecast = self.forecasting.fit(
                ts=ts_train,
                ext_vars=ext_train
            ).forecast(
                steps=h,
                ext_vars=ext_future
            )
            assert len(ts_forecast) == h
            self.forecasts[iteration] = ts_forecast

            self._compute_metrics(ts_test, ts_forecast)
            if plot_segments:
                self._show_plots(iteration, ts_left, ts_train, ts_test, ts_right, ts_forecast)
            if plot_residuals:
                self._plot_residuals(ts_forecast, ts_test)
            if plot_pulls:
                self._plot_pulls(ts_forecast, ts_test)

        self.metrics_data = pandas.DataFrame(self.metrics_results)
        return self

    def get_metrics(self):
        return self.metrics_data
