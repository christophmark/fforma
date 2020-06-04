import pandas as pd
import numpy as np
import multiprocessing as mp
import lightgbm as lgb

import copy

from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax
from tsfeatures import tsfeatures
from math import isclose
from fforma.utils_input import _check_valid_df, _check_same_type, _check_passed_dfs, _check_valid_columns
from fforma.utils_models import _train_lightgbm, _train_lightgbm_cv, _train_lightgbm_grid_search



class FFORMA:

    def __init__(self, objective='FFORMA', verbose_eval=True,
                 early_stopping_rounds=10,
                 params=None,
                 param_grid=None,
                 use_cv=False, nfolds=5,
                 greedy_search=False,
                 threads=None, seed=260294):
        """ Feature-based Forecast Model Averaging.

        Python Implementation of FFORMA.

        Parameters
        ----------
        xgb_params:
            Parameters for the xgboost
        obj: str
            Type of error to calculate
        -----
        ** References: **
        <https://robjhyndman.com/publications/fforma/>
        """
        self.dict_obj = {'FFORMA': (self.fforma_objective, self.fforma_loss),
                         'FFORMADL': (self.fformadl_objective, self.fformadl_loss)}

        fobj, feval = self.dict_obj.get(objective, (None, None))
        self.objective, self.greedy_search = objective, greedy_search

        if threads is None:
            threads = mp.cpu_count() - 1

        init_params = {
            'objective': 'multiclass',
            'nthread': threads,
            'seed': seed
        }

        if params:
            train_params = {**params, **init_params}
        else:
            train_params = {'n_estimators': 100}
            train_params = {**train_params, **init_params}


        if param_grid is not None:
            folds = lambda holdout_feats, best_models: StratifiedKFold(n_splits=nfolds).split(holdout_feats, best_models)

            self._train = lambda holdout_feats, best_models: _train_lightgbm_grid_search(holdout_feats, best_models,
                                                                                use_cv, init_params, param_grid, fobj, feval,
                                                                                early_stopping_rounds, verbose_eval,
                                                                                seed, folds)
        elif use_cv:
            folds = lambda holdout_feats, best_models: StratifiedKFold(n_splits=nfolds).split(holdout_feats, best_models)

            self._train = lambda holdout_feats, best_models: _train_lightgbm_cv(holdout_feats, best_models,
                                                                                train_params, fobj, feval,
                                                                                early_stopping_rounds, verbose_eval,
                                                                                seed, folds)
        else:
            self._train = lambda holdout_feats, best_models: _train_lightgbm(holdout_feats, best_models,
                                                                             train_params, fobj, feval,
                                                                             early_stopping_rounds, verbose_eval,
                                                                             seed)
        self._fitted = False

    def _tsfeatures(self, y_train_df, y_val_df, freq):
        #TODO receive panel of freq
        complete_data = pd.concat([y_train_df, y_test_df.filter(items=['unique_id', 'ds', 'y'])])
        holdout_feats = tsfeatures(y_train_df)
        feats = tsfeatures(complete_data)

        return feats, holdout_feats

    # Objective function for lgb
    def fforma_objective(self, predt: np.ndarray, dtrain) -> (np.ndarray, np.ndarray):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        preds_transformed = softmax(preds, axis=1)

        weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))

        grad = preds_transformed*(self.contribution_to_error[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_error[y,:]*preds_transformed*(1.0-preds_transformed) - grad*preds_transformed
        #hess = grad*(1 - 2*preds_transformed)
        return grad.flatten('F'), hess.flatten('F')

    def fforma_loss(self, predt: np.ndarray, dtrain) -> (str, float):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        #for lightgbm
        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')
        #lightgbm uses margins!
        preds_transformed = softmax(preds, axis=1)
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1)
        fforma_loss = weighted_avg_loss_func.mean()

        return 'FFORMA-loss', fforma_loss, False

    def fit(self, y_train_df=None, y_val_df=None,
            val_periods=None,
            errors=None, holdout_feats=None,
            feats=None, freq=None, base_model=None,
            sorted_data=False, weights=None):
        """
        y_train_df: pandas df
            panel with columns unique_id, ds, y
        y_val_df: pandas df
            panel with columns unique_id, ds, y, {model} for each model to ensemble
        val_periods: int or pandas df
            int: number of val periods
            pandas df: panel with columns unique_id, val_periods
        """

        if (errors is None) and (feats is None):
            assert (y_train_df is not None) and (y_val_df is not None), "you must provide a y_train_df and y_val_df"
            is_pandas_df = _check_passed_dfs(y_train_df, y_val_df_)

            if not sorted_data:
                if is_pandas_df:
                    y_train_df = y_train_df.sort_values(['unique_id', 'ds'])
                    y_val_df = y_val_df.sort_values(['unique_id', 'ds'])
                else:
                    y_train_df = y_train_df.sort_index()
                    y_val_df = y_val_df.sort_index()

        if errors is None:
            pass
            #calculate contribution_to_error(y_train_df, y_val_df)
        else:
            _check_valid_columns(errors, cols=['unique_id'], cols_index=['unique_id'])

            best_models_count = errors.idxmin(axis=1).value_counts()
            best_models_count = pd.Series(best_models_count, index=errors.columns)
            loser_models = best_models_count[best_models_count.isna()].index.to_list()

            if len(loser_models) > 0:
                print('Models {} never win.'.format(' '.join(loser_models)))
                print('Removing it...\n')
                errors = errors.copy().drop(columns=loser_models)

            self.contribution_to_error = errors.values
            best_models = self.contribution_to_error.argmin(axis=1)


        if feats is None:
            feats, holdout_feats = self._tsfeatures(y_train_df, y_val_df, freq)
        else:
            assert holdout_feats is not None, "when passing feats you must provide holdout feats"

        self.lgb = self._train(holdout_feats, best_models)

        raw_score_ = self.lgb.predict(feats, raw_score=True)
        self.raw_score_ = pd.DataFrame(raw_score_,
                                       index=feats.index,
                                       columns=errors.columns)

        weights = softmax(raw_score_, axis=1)
        self.weights_ = pd.DataFrame(weights,
                                     index=feats.index,
                                     columns=errors.columns)

        if self.greedy_search:
            performance = self.lgb.best_score['valid_1']['FFORMA-loss']
            improvement = True
            errors = copy.deepcopy(errors)
            print(f'\nInitial performance: {performance}\n')
            while improvement and errors.shape[1]>2:
                model_to_remove = self.weights_.mean().nsmallest(1).index
                print(f'Removing {model_to_remove}\n')
                errors = errors.drop(columns=model_to_remove)
                self.contribution_to_error = errors.values
                best_models = self.contribution_to_error.argmin(axis=1)

                new_lgb = self._train(holdout_feats, best_models)
                performance_new_lgb = new_lgb.best_score['valid_1']['FFORMA-loss']
                better_model = performance_new_lgb <= performance
                if not better_model:
                    print(f'\nImprovement not reached: {performance_new_lgb}')
                    print('stopping greedy_search')
                    improvement = False
                else:
                    performance = performance_new_lgb
                    print(f'\nReached better performance {performance}\n')
                    self.lgb = new_lgb

                    raw_score_ = self.lgb.predict(feats, raw_score=True)
                    self.raw_score_ = pd.DataFrame(raw_score_,
                                                   index=feats.index,
                                                   columns=errors.columns)

                    weights = softmax(raw_score_, axis=1)
                    self.weights_ = pd.DataFrame(weights,
                                                 index=feats.index,
                                                 columns=errors.columns)
        self._fitted = True


    def predict(self, y_hat_df, fforms=False):
        """
        Parameters
        ----------
        y_hat_df: pandas df
            panel with columns unique_id, ds, {model} for each model to ensemble
        """
        assert self._fitted, "Model not fitted yet"

        if fforms:
            weights = (self.weights_.div(self.weights_.max(axis=1), axis=0) == 1)*1
            name = 'fforms_prediction'
        else:
            weights = self.weights_
            name = 'fforma_prediction'
        fforma_preds = weights * y_hat_df
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = name
        preds = pd.concat([y_hat_df, fforma_preds], axis=1)

        return preds
