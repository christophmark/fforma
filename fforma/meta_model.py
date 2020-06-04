#!/usr/bin/env python
# coding: utf-8

import numpy as np

from copy import deepcopy
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted
from ESRNN.utils_evaluation import smape, mase, evaluate_panel
from collections import ChainMap
from functools import partial
import dask
from dask.diagnostics import ProgressBar
from rpy2.robjects import pandas2ri


class MetaModels:
    """
    Train models to ensemble.

    Parameters
    ----------
    models: dict
        Dictionary of models to train. Ej {'ARIMA': ARIMA()}
    """

    def __init__(self, models, dask_client=None):
        self.models = models
        self.dask_client = dask_client

    def fit(self, y_panel_df):
        """For each time series fit each model in models.

        y_panel_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        """
        if self.dask_client is not None:
            fit_single_ts_p = partial(fit_single_ts, models=self.models)
            futures = self.dask_client.map(fit_single_ts_p, y_panel_df.groupby('unique_id'))
            fitted_models = self.dask_client.gather(futures)
        else:
            fitted_models = [fit_single_ts(x, self.models) for x in tqdm(y_panel_df.groupby('unique_id'))]

        self.fitted_models_ = dict(ChainMap(*fitted_models))

        return self

    def predict(self, y_hat_df):
        """Predict each model for each time series.

        y_hat_df: pandas df
            Pandas DataFrame with columns ['unique_id', 'ds']
        """
        check_is_fitted(self, 'fitted_models_')

        y_hat_df = deepcopy(y_hat_df)

        for col in self.models.keys():
            y_hat_df[col] = None

        for key, value in self.fitted_models_.items():
            filter_key = y_hat_df['unique_id'] == key
            y_df = y_hat_df.loc[filter_key]
            for model_name, model in value.items():
                y_hat_df.loc[filter_key, model_name] = model.predict(len(y_df))

        return y_hat_df

################################################################################
########## UTILS FOR FFORMA FLOW
###############################################################################

def fit_single_model(model, y):
    this_model = deepcopy(model)
    fitted_model = this_model.fit(y)

    return fitted_model

def fit_single_ts(x, models):
    idx, df = x
    fitted_ts = {
        name_model: fit_single_model(model, df['y'].values) for name_model, model in models.items()
    }

    fitted_ts = {idx: fitted_ts}

    return fitted_ts

def temp_holdout(y_panel_df, val_periods):
    """Splits the data in train and validation sets.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']

    Returns
    -------
    Tuple
        - train: pandas df
        - val: pandas df
    """
    val = y_panel_df.groupby('unique_id').tail(val_periods)
    train = y_panel_df.groupby('unique_id').apply(lambda df: df.head(-val_periods)).reset_index(drop=True)

    return train, val

def calc_errors(y_panel_df, y_insample_df, seasonality, benchmark_model='Naive2'):
    """Calculates OWA of each time series
    usign benchmark_model as benchmark.

    Parameters
    ----------
    y_panel_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
    y_insample_df: pandas df
        Pandas DataFrame with columns ['unique_id', 'ds', 'y']
        Train set.
    seasonality: int
        Frequency of the time seires.
    benchmark_model: str
        Column name of the benchmark model.

    Returns
    -------
    Pandas DataFrame
        OWA errors for each time series and each model.
    """

    assert benchmark_model in y_panel_df.columns

    y_panel = y_panel_df[['unique_id', 'ds', 'y']]
    y_hat_panel_fun = lambda model_name: y_panel_df[['unique_id', 'ds', model_name]].rename(columns={model_name: 'y_hat'})

    model_names = set(y_panel_df.columns) - set(y_panel.columns)

    errors_smape = y_panel[['unique_id']].drop_duplicates().reset_index(drop=True)
    errors_mase = errors_smape.copy()

    for model_name in model_names:
        errors_smape[model_name] = None
        errors_mase[model_name] = None
        y_hat_panel = y_hat_panel_fun(model_name)

        errors_smape[model_name] = evaluate_panel(y_panel, y_hat_panel, smape)
        errors_mase[model_name] = evaluate_panel(y_panel, y_hat_panel, mase, y_insample_df, seasonality)

    mean_smape_benchmark = errors_smape[benchmark_model].mean()
    mean_mase_benchmark = errors_mase[benchmark_model].mean()

    errors_smape = errors_smape.drop(columns=benchmark_model).set_index('unique_id')
    errors_mase = errors_mase.drop(columns=benchmark_model).set_index('unique_id')

    errors = errors_smape/mean_mase_benchmark + errors_mase/mean_smape_benchmark
    errors = 0.5*errors
    errors = errors

    return errors

def get_prediction_panel(y_panel_df, h, freq):
    """Construct panel to use with
    predict method.
    """
    df = y_complete_train_df[['unique_id', 'ds']].groupby('unique_id').max().reset_index()

    predict_panel = []
    for idx, df in df.groupby('unique_id'):
        date = df['ds'].values.item()
        unique_id = df['unique_id'].values.item()

        date_range = pd.date_range(date, periods=4, freq='D')
        df_ds = pd.DataFrame.from_dict({'ds': date_range})
        df_ds['unique_id'] = unique_id
        predict_panel.append(df_ds[['unique_id', 'ds']])

    predict_panel = pd.concat(predict_panel)

    return predict_panel
