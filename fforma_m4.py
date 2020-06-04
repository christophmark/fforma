#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from fforma import FFORMA
from fforma.r_models import (
    ARIMA,
    ETS,
    ThetaF,
    Naive,
    SeasonalNaive
)
from fforma.meta_model import (
    MetaModels,
    temp_holdout,
    calc_errors,
    get_prediction_panel
)
from ESRNN.utils_evaluation import Naive2
from ESRNN.m4_data import prepare_m4_data, seas_dict
from tsfeatures import tsfeatures


def prepare_to_train_fforma(dataset, validation_periods, seasonality):

    X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset, './data', 100)

    # Preparing errors
    y_holdout_train_df, y_val_df = temp_holdout(y_train_df, validation_periods)
    meta_models = {
        #'ARIMA': ARIMA(freq=seasonality, stepwise=False, approximation=False),
        'ETS': ETS(freq=seasonality),
        'ThetaF': ThetaF(freq=seasonality),
        'Naive': Naive(freq=seasonality),
        'SeasonalNaive': SeasonalNaive(freq=seasonality),
        'Naive2': Naive2(seasonality=seasonality)
    }
    validation_meta_models = MetaModels(meta_models)
    validation_meta_models.fit(train)
    prediction_validation_meta_models = validation_meta_models.predict(y_val_df)

    #Calculating errors
    errors = calc_errors(prediction_validation_meta_models, y_holdout_train_df, seasonality)

    #Calculating features
    features = tsfeatures(y_holdout_train_df, seasonality)

    #Calculating actual predictins
    meta_models = MetaModels(meta_models)
    meta_models.fit(y_train_df)

    predictions = meta_models.predict(y_test_df[['unique_id', 'ds']]))

    return errors, features, predictions


def main():
    complete_errors, complete_features, complete_predictions = [], [], []

    for dataset in ['Weekly']: #'Daily', etc
        validation_periods = seas_dict[dataset]['output_size']
        seasonality = seas_dict[dataset]['seasonality']
        errors, features, predictions = prepare_to_train_fforma(dataset, validation_periods, seasonality)

        complete_errors.append(errors)
        complete_features.append(features)
        complete_predictions.append(predictions)

    complete_errors = pd.concat(complete_errors)
    complete_features = pd.concat(complete_features)
    complete_predictions = pd.concat(complete_predictions)

    #Training fforma

    # optimal params by hyndman
    optimal_params = {'n_estimators': 94,
                      'eta': 0.58,
                      'max_depth': 14,
                      'subsample': 0.92,
                      'colsample_bytree': 0.77}
    fforma = FFORMA(params=optimal_params)
    fforma.fit(errors=complete_errors,
               holdout_feats=complete_features,
               feats=complete_features)

    fforma_predictions = fforma.predict(complete_predictions)

    #evaluate predictions


if __name__=='__main__':
    main()
