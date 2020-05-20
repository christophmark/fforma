import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import copy
import multiprocessing as mp
import pickle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import tqdm
import numpy as np
from fforma.benchmarks import *
from tsfeatures import tsfeatures
import lightgbm as lgb

from fforma.utils_input import CheckInput
from fforma.utils_models import LightGBM, XGBoost

from math import isclose



class FFORMA(CheckInput, LightGBM, XGBoost):
    def __init__(self, objective='fforma', verbose_eval=True,
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
        self.dict_obj = {'FFORMA': (self.fforma_objective,
                                    lambda predt, dtrain: [self.fforma_loss(predt, dtrain),
                                                           self.fformadl_loss(predt, dtrain)]),
                         'FFORMADL': (self.fformadl_objective, self.fformadl_loss)}


        fobj, feval = self.dict_obj.get(objective, (None, self.fforma_loss))
        self.objective = objective
        self.greedy_search = greedy_search



        # nthreads params
        if threads is None:
            threads = mp.cpu_count() - 1

        init_params = {
            'objective': 'multiclass',
            #'num_class': self.contribution_to_error.shape[1],
            'nthread': threads,
            'silent': 1,
            'seed': seed
        }

        if params:
            train_params = {**params, **init_params}
        else:
            train_params = {'n_estimators': 100}
            train_params = {**train_params, **init_params}


        if param_grid is not None:
            pass
        elif use_cv:
            folds = lambda holdout_feats, best_models: StratifiedKFold(n_splits=nfolds).split(holdout_feats, best_models)

            self._train = lambda holdout_feats, best_models: self._train_lightgbm_cv(holdout_feats, best_models,
                                                                                     train_params, fobj, feval,
                                                                                     early_stopping_rounds, verbose_eval,
                                                                                     seed, folds)
        else:
            self._train = lambda holdout_feats, best_models: self._train_lightgbm(holdout_feats, best_models,
                                                                                  train_params, fobj, feval,
                                                                                  early_stopping_rounds, verbose_eval,
                                                                                  seed)


    def _tsfeatures(self, y_train_df, y_val_df, freq):
        #TODO receive panel of freq
        complete_data = pd.concat([y_train_df, y_test_df.filter(items=['unique_id', 'ds', 'y'])])
        holdout_feats = tsfeatures(y_train_df)
        feats = tsfeatures(complete_data)

        return feats, holdout_feats

    def fformadl_objective(self, predt: np.ndarray, dtrain) -> (np.ndarray, np.ndarray):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        y_index = self.indices[y]
        y_val_to_ensemble = self.y_val_to_ensemble.loc[y_index]
        y_val = self.y_val.loc[y_index]

        y_train_index = y_val.index.get_level_values('unique_id')

        preds = np.reshape(predt,
                          self.contribution_to_error[y, :].shape,
                          order='F')

        #lightgbm uses margins!
        preds = softmax(preds, axis=1)
        preds_transformed = pd.DataFrame(preds, index=y_index,
                                         columns=y_val_to_ensemble.columns)


        y_val_pred = (preds_transformed*y_val_to_ensemble).sum(axis=1)
        diff_actual_y_val = y_val.sub(y_val_pred, axis=0)
        diff_ensemble_y_val = y_val_to_ensemble.sub(y_val_pred, axis=0)

        # Gradient
        #grad_pred_model = preds_transformed.mul(diff_ensemble_y_val, axis=0)
        grad_pred_model = preds_transformed.mul(diff_ensemble_y_val, axis=0)


        #grad_mse = (-grad_pred_model).mul(diff_actual_y_val, axis=0)
        grad_mse = (-grad_pred_model).mul(diff_actual_y_val, axis=0)
        grad_mse = grad_mse.groupby(y_train_index).mean()
        grad_mse = 2*grad_mse

        if self.loss_weights is not None:
            grad_mse = grad_mse.mul(self.loss_weights.loc[y_index], axis=0)

        #Hessian
        #hess_pred_model = (1-preds_transformed).mul(diff_ensemble_y_val, axis=0) - grad_pred_model
        hess_pred_model = (1-preds_transformed).mul(y_val_pred, axis=0) - grad_pred_model
        hess_pred_model = preds_transformed*hess_pred_model
        hess_mse = (-hess_pred_model).mul(diff_ensemble_y_val, axis=0)

        second_term_hess_mse = grad_pred_model.pow(2)

        hess_mse = hess_mse + second_term_hess_mse
        hess_mse = hess_mse.groupby(y_train_index).mean()
        hess_mse = 2*hess_mse

        if self.loss_weights is not None:
            hess_mse = hess_mse.mul(self.loss_weights.loc[y_index], axis=0)

        grad = grad_mse.values
        hess = hess_mse.values


        return grad.flatten('F'), hess.flatten('F')


    def fformadl_loss(self, predt: np.ndarray, dtrain) -> (str, float):
        '''
        Compute...
        '''
        #print(predt.sum(axis=1).mean())
        y = dtrain.get_label().astype(int)
        y_index = self.indices[y]
        y_val_to_ensemble = self.y_val_to_ensemble.loc[y_index]
        y_val = self.y_val.loc[y_index]

        preds = np.reshape(predt,
                           (len(y_index), len(y_val_to_ensemble.columns)),
                           order='F')
        #lightgbm uses margins!
        preds = softmax(preds, axis=1)
        preds_transformed = pd.DataFrame(preds, index=y_index,
                                         columns=y_val_to_ensemble.columns)

        y_val_pred = (preds_transformed*y_val_to_ensemble).sum(axis=1)
        mse = y_val.sub(y_val_pred, axis=0)
        mse = mse.pow(2)

        #print(mse)
        mse = mse.groupby(mse.index.get_level_values('unique_id')).mean().pow(1/2)
        #print(mse)
        #mse = 0.5*np.log(mse) #+ 0.5*np.log(self.loss_weights.loc[y_index])#
        if self.loss_weights is not None:
            mse = mse.mul(self.loss_weights.loc[y_index], axis=0)#.pow(1/2)

        #print(mse.shape)
            mse = mse.values.sum()
        else:
            mse = mse.values.mean()

        return 'FFORMADL-loss', mse, False

    # Objective function for xgb
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
            feats=None, weights=None,
            freq=None, base_model=None,
            sorted_data=False, init_score=None):
        """
        y_train_df: pandas df
            panel with columns unique_id, ds, y
        y_val_df: pandas df
            panel with columns unique_id, ds, y, {model} for each model to ensemble
        val_periods: int or pandas df
            int: number of val periods
            pandas df: panel with columns unique_id, val_periods
        """

        self.loss_weights = weights

        if (errors is None) and (feats is None):
            assert (y_train_df is not None) and (y_val_df is not None), "you must provide a y_train_df and y_val_df"
            is_pandas_df = self._check_passed_dfs(y_train_df, y_val_df_)

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
            self._check_valid_columns(errors, cols=['unique_id'], cols_index=['unique_id'])
            self.contribution_to_error = errors.values
            best_models = self.contribution_to_error.argmin(axis=1)


        if feats is None:
            feats, holdout_feats = self._tsfeatures(y_train_df, y_val_df, freq)
        else:
            assert holdout_feats is not None, "when passing feats you must provide holdout feats"
            self._check_valid_columns(feats, cols=['unique_id'], cols_index=['unique_id'])

        self.indices = holdout_feats.index.get_level_values('unique_id')

        if self.objective == 'FFORMADL':
            assert y_val_df is not None, 'FFORMADL needs y_val_df'

        self.y_val_to_ensemble = y_val_df.drop(columns='y')
        self.y_val = y_val_df['y']


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
            performance = self.lgb.best_score['valid_1']['multi_logloss']
            improvement = True
            errors = copy.deepcopy(errors)
            print(f'\nInitial performance: {performance}\n')
            while improvement and errors.shape[1]>2:
                print(errors.shape)
                model_to_remove = self.raw_score_.mean().idxmin()
                print(f'Removing {model_to_remove}\n')
                errors = errors.drop(columns=model_to_remove)
                self.contribution_to_error = errors.values
                self.y_val_to_ensemble = self.y_val_to_ensemble.drop(columns=model_to_remove)
                best_models = self.contribution_to_error.argmin(axis=1)

                new_lgb = self._train(holdout_feats, best_models)
                performance_new_lgb = new_lgb.best_score['valid_1']['multi_logloss']
                better_model = performance_new_lgb < performance
                if not better_model:
                    print('\nImprovement not reached, stopping greedy_search')
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

        return self


    def predict(self, y_hat_df, fforms=False):
        """
        Parameters
        ----------
        y_hat_df: pandas df
            panel with columns unique_id, ds, {model} for each model to ensemble
        """
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
