import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import copy
import multiprocessing as mp
from sklearn.model_selection import train_test_split
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



class FForma(CheckInput, LightGBM, XGBoost):
    def __init__(self, objective='fforma', verbose_eval=True,
                 early_stopping_rounds=10,
                 params=None,
                 param_grid=None,
                 use_cv=False, nfolds=5,
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
        self.dict_obj = {'fforma': (self.fforma_objective, self.fforma_loss),
                         'fformadl': (self.fformadl_objective, self.fformadl_loss)}

        self.verbose_eval = verbose_eval
        self.early_stopping_rounds = early_stopping_rounds

        self.params = params
        self.grid_search = False
        self.param_grid = param_grid

        if self.param_grid is not None:
            self.grid_search = True

        self.objective = objective

        self.use_cv = use_cv
        self.nfolds = nfolds

        # nthreads params
        if threads is None:
            self.threads = mp.cpu_count() - 1

        self.seed = seed

        self.fobj, self.feval = self.dict_obj[self.objective]

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
        preds_transformed = pd.DataFrame(preds, index=self.holdout_feats_.loc[y_index].index,
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
            self.best_models_ = self.contribution_to_error.argmin(axis=1)


        if feats is None:
            self.feats_, self.holdout_feats_ = self._tsfeatures(y_train_df, y_val_df, freq)
        else:
            assert holdout_feats is not None, "when passing feats you must provide holdout feats"
            self._check_valid_columns(feats, cols=['unique_id'], cols_index=['unique_id'])
            self.feats_, self.holdout_feats_ = feats, holdout_feats


        # Train-validation sets for XGBoost
        self.holdout_feats_train_, self.holdout_feats_val_, \
            self.best_models_train_, \
            self.best_models_val_, \
            self.indices_train_, \
            self.indices_val_ = train_test_split(self.holdout_feats_,
                                                 self.best_models_,
                                                 np.arange(self.holdout_feats_.shape[0]),
                                                 random_state=self.seed,
                                                 stratify=self.best_models_)

        self.index_train = self.holdout_feats_train_.index.get_level_values('unique_id')
        self.index_val = self.holdout_feats_val_.index.get_level_values('unique_id')

        self.indices = self.holdout_feats_.index.get_level_values('unique_id')
        self.index_train = self.holdout_feats_train_.index.get_level_values('unique_id')
        self.index_val = self.holdout_feats_val_.index.get_level_values('unique_id')


        if self.objective == 'fformadl':
            assert y_val_df is not None, 'FFORMADL needs y_val_df'

            self.y_val_to_ensemble = y_val_df.drop(columns='y')
            self.y_val = y_val_df['y']
            self.y_val_to_ensemble_train = self.y_val_to_ensemble.loc[self.index_train]
            self.y_val_train = self.y_val.loc[self.index_train]
            self.index_y_train = self.y_val_to_ensemble_train.index.get_level_values('unique_id')

        if self.objective in self.dict_obj.keys():
            if init_score is not None:
                self.dtrain = lgb.Dataset(data=self.holdout_feats_train_, label=self.indices_train_,
                                          init_score=init_score.loc[self.index_train].values.flatten('F'))
            else:
                self.dtrain = lgb.Dataset(data=self.holdout_feats_train_, label=self.indices_train_)
            self.dvalid = lgb.Dataset(data=self.holdout_feats_val_, label=self.indices_val_)
        else:
            self.dtrain = lgb.Dataset(data=self.holdout_feats_train_, label=self.best_models_train_)
            self.dvalid = lgb.Dataset(data=self.holdout_feats_val_, label=self.best_models_val_)


        self.init_params = {
            'objective': 'multiclass',
            'num_class': self.contribution_to_error.shape[1],
            'nthread': self.threads,
            'silent': 1,
            'seed': self.seed
        }

        if self.params:
            params = {**self.params, **self.init_params}
        else:
            params = {'n_estimators': 100}
            params = {**params, **self.init_params}
        # else:
        #     # From http://htmlpreview.github.io/?https://github.com/robjhyndman/M4metalearning/blob/master/docs/M4_methodology.html
        #     params = {
        #         'n_estimators': 94,
        #         'eta': 0.58,
        #         'max_depth': 14,
        #         'subsample': 0.92,
        #         'colsample_bytree': 0.77
        #     }
        #     params = {**params, **self.init_params}


        #self.xgb = self._train_xgboost(params)
        if self.grid_search:
            self.verbose_eval_grid = self.verbose_eval
            self.verbose_eval = False
            self.lgb = self._search_optimal_params_lightgbm(self.param_grid)
        else:
            self.lgb = self._train_lightgbm(params)

        #weights = self.xgb.predict(xgb.DMatrix(self.feats_))
        raw_score_ = self.lgb.predict(self.feats_, raw_score=True)
        self.raw_score_ = pd.DataFrame(raw_score_,
                                       index=self.feats_.index,
                                       columns=errors.columns)

        weights = softmax(raw_score_, axis=1)
        self.weights_ = pd.DataFrame(weights,
                                     index=self.feats_.index,
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
