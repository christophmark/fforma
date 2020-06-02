#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import rpy2.robjects as robjects

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import IntVector

forecast = importr('forecast')
stats = importr('stats')
base = importr('base')

ts = robjects.r('ts')
msts = robjects.r('msts')


pandas2ri.activate()

def forecast_object_to_dict(forecast_object):
    """Transform forecast_object into a python dictionary."""
    dict_ = zip(forecast_object.names,
                list(forecast_object))
    dict_ = dict(dict_)

    return dict_

def get_forecast(fitted_model, h):
    """Calculate forecast from a fitted model."""
    y_hat = forecast.forecast(fitted_model, h=h)
    y_hat = forecast_object_to_dict(y_hat)
    y_hat = y_hat['mean']

    return y_hat

class ForecastModel:
    """Wrapper for models in the R package _forecast_ that returns a model.

    Parameters
    ----------
    model: rpy2 object
        Model from forecat rpy2 api. Ej. forecast.auto_arima
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities.
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.model = lambda y: model(y, **kwargs)
        if isinstance(freq, int):
            self.freq = freq
        else:
            self.freq = IntVector(freq)

    def fit(self, X, y):
        y_ts = msts(y.values, seasonal_periods=self.freq)
        self.fitted_model_ = self.model(y_ts)

        return self

    def predict(self, X):
        check_is_fitted(self, 'fitted_model_')
        h = X.shape[0]

        y_hat = get_forecast(self.fitted_model_, h)

        return y_hat

class ForecastObject:
    """Wrapper for models in the R package _forecast_ that returns a forecast object.

    Parameters
    ----------
    model: rpy2 object
        Model from forecat rpy2 api. Ej. forecast.auto_arima
    freq: int or iterable
        Frequency of the time series.
        Can be multiple seasonalities.
    kwargs:
        Arguments of the model function.
    """

    def __init__(self, model, freq, **kwargs):
        self.model = lambda y, h: model(y, h=h, **kwargs)
        if isinstance(freq, int):
            self.freq = freq
        else:
            self.freq = IntVector(freq)

    def fit(self, X, y):
        self.y_ts_ = msts(y.values, seasonal_periods=self.freq)

        return self

    def predict(self, X):
        check_is_fitted(self, 'y_ts_')
        h = X.shape[0]

        fitted_model = self.model(self.y_ts_, h)

        y_hat = get_forecast(fitted_model, h)

        return y_hat

class ARIMA(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::auto.arima from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.auto_arima, freq=freq, **kwargs)

class ETS(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::ets from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.ets, freq=freq, **kwargs)

class NNETAR(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::nnetar from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.nnetar, freq=freq, **kwargs)

class TBATS(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::tbats from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.tbats, freq=freq, **kwargs)

class STLM(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::stlm from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.stlm, freq=freq, **kwargs)

class RandomWalk(ForecastModel, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::rwf from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.rwf, freq=freq, **kwargs)

class ThetaF(ForecastObject, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::thetaf from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.thetaf, freq=freq, **kwargs)

class Naive(ForecastObject, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::naive from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.naive, freq=freq, **kwargs)

class SeasonalNaive(ForecastObject, BaseEstimator, RegressorMixin):
    """Wrapper of forecast::snaive from R.

    Parameters
    ----------
    freq: int
        Frequency of the time series.
    """

    def __init__(self, freq=7, **kwargs):
        super().__init__(model=forecast.snaive, freq=freq, **kwargs)
