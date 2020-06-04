import pandas as pd
import lightgbm as lgb
import numpy as np

import copy

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def _train_lightgbm(holdout_feats, best_models,
                    params, fobj, feval,
                    early_stopping_rounds,
                    verbose_eval, seed):

    holdout_feats_train, holdout_feats_val, \
        best_models_train, \
        best_models_val, \
        indices_train, \
        indices_val = train_test_split(holdout_feats,
                                       best_models,
                                       np.arange(holdout_feats.shape[0]),
                                       random_state=seed,
                                       stratify=best_models)

    params = copy.deepcopy(params)
    num_round = int(params.pop('n_estimators', 100))

    params['num_class'] = len(np.unique(best_models))

    if fobj is not None:

        dtrain = lgb.Dataset(data=holdout_feats_train, label=indices_train)
        dvalid = lgb.Dataset(data=holdout_feats_val, label=indices_val)
        valid_sets = [dtrain, dvalid]

        gbm_model = lgb.train(
            params=params,
            train_set=dtrain,
            fobj=fobj,
            num_boost_round=num_round,
            feval=feval,
            valid_sets=valid_sets,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
    else:

        dtrain = lgb.Dataset(data=holdout_feats_train, label=best_models_train)
        dvalid = lgb.Dataset(data=holdout_feats_val, label=best_models_val)
        valid_sets = [dtrain, dvalid]

        gbm_model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=num_round,
            valid_sets=valid_sets,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )


    return gbm_model

def _train_lightgbm_cv(holdout_feats, best_models,
                       params, fobj, feval,
                       early_stopping_rounds,
                       verbose_eval, seed,
                       folds, train_model=True):

    params = copy.deepcopy(params)
    num_round = int(params.pop('n_estimators', 100))

    params['num_class'] = len(np.unique(best_models))

    if fobj is not None:
        indices = np.arange(holdout_feats.shape[0])
        dtrain = lgb.Dataset(data=holdout_feats, label=indices)

        gbm_model = lgb.cv(
            params=params,
            train_set=dtrain,
            fobj=fobj,
            num_boost_round=num_round,
            feval=feval,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval = verbose_eval,
            folds=folds(holdout_feats, best_models)
        )
    else:
        dtrain = lgb.Dataset(data=holdout_feats, label=best_models)

        gbm_model = lgb.cv(
            params=params,
            train_set=dtrain,
            num_boost_round=num_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            folds=folds(holdout_feats, best_models)
        )

    optimal_rounds = len(gbm_model[list(gbm_model.keys())[0]])
    best_performance = gbm_model[list(gbm_model.keys())[0]][-1]

    if train_model:
        params['n_estimators'] = optimal_rounds

        optimal_gbm_model = _train_lightgbm(holdout_feats, best_models,
                                            params, fobj, feval,
                                            early_stopping_rounds,
                                            verbose_eval, seed)

        return optimal_gbm_model

    return optimal_rounds, best_performance

def _train_lightgbm_grid_search(holdout_feats, best_models,
                                use_cv, init_params,
                                param_grid, fobj, feval,
                                early_stopping_rounds,
                                verbose_eval, seed,
                                folds):

    best_params = {}
    best_performance = np.inf

    pbar = tqdm(ParameterGrid(param_grid))
    pbar.set_description('Best performance: ??')

    for params in pbar:

        params = {**params, **init_params}

        if use_cv:
            num_round, performance =  _train_lightgbm_cv(holdout_feats, best_models,
                                                         params, fobj, feval,
                                                         early_stopping_rounds,
                                                         False, seed,
                                                         folds, train_model=False)
        else:
            gbm_model = _train_lightgbm(holdout_feats, best_models,
                                        params, fobj, feval,
                                        early_stopping_rounds,
                                        False, seed)
            performance = list(gbm_model.best_score['valid_1'].values())[0]
            num_round = gbm_model.best_iteration

        if performance < best_performance:
            #Updating  best performance
            pbar.set_description('Best performance: {}'.format(performance))
            #Updating bars
            best_params = params
            best_performance = performance
            best_params['n_estimators'] = num_round


    optimal_gbm_model = _train_lightgbm(holdout_feats, best_models,
                                        best_params, fobj, feval,
                                        early_stopping_rounds,
                                        False, seed)

    return optimal_gbm_model
