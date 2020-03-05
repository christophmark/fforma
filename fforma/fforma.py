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


class FForma:
    def __init__(self, y_train_df=None, y_val_df=None, y_hat_val_df=None, frcy=None, max_evals=100):
        """ Feature-based Forecast Model Averaging.

        Python Implementation of FFORMA.

        Parameters
        ----------
        y_train_df: pandas df
            panel with columns unique_id, ds, y
        y_val_df: pandas df
            panel with columns unique_id, ds, y
        y_hat_val_df: pandas df
            panel with columns unique_id, ds, y_{model} for each model to ensemble
        frcy: int
            frequency of time series
        max_evals: int
        -----
        ** References: **
        <https://robjhyndman.com/publications/fforma/>
        """

        self.ts_list = [df['y'].values for idx, df in y_train_df.groupby('unique_id')]
        self.ts_val_list = [df['y'].values for idx, df in y_val_df.groupby('unique_id')]

        self.models = y_hat_val_df.columns[y_hat_val_df.columns.str.contains('y_')]

        assert len(self.models) > 1, 'FFORMA ensambles more than one model'

        self.ts_hat_val_list = [[df[col].values for col in self.models] for idx, df in y_hat_val_df.groupby('unique_id')]
        self.frcy = frcy

        print('Setting model')
        self.X_feats, self.y_best_model, self.contribution_to_error = self._prepare_to_train(
            self.ts_list, self.ts_val_list, self.ts_hat_val_list, self.frcy
        )

        self.n_models = len(self.models)

        self.max_evals = max_evals

        self.dict_obj = {'fforma': self.fforma_objective, 'softmax': self.softmax_objective}

    def _prepare_to_train(self, ts_list, ts_val_list, ts_hat_val_list, frcy):
        '''

        Returns tsfeatures of ts, best model for each time series
        and contribution to owa
        '''
        #n_models =
        print('Computing contribution to owa')
        contribution_to_owa = self.contribution_to_owa(
            ts_list, ts_val_list, ts_hat_val_list, frcy
        )
        best_models = contribution_to_owa.argmin(axis=1)
        print('Computing features')
        ts_features = tsfeatures(ts_list, frcy=frcy)

        return ts_features, best_models, contribution_to_owa

    def contribution_to_owa(self, ts_list, ts_val_list, ts_hat_val_list, frcy):
        print('Training NAIVE2')
        len_val = len(ts_val_list[0])
        ts_val_naive2 = [Naive2().fit(ts, frcy).predict(len_val) for ts in tqdm.tqdm(ts_list)]

        print('Calculating errors')
        # Smape
        smape_errors = np.array([
            np.array(
                [self.smape(ts_val, pred) for pred in ts_hat_val]
            ) for ts_val, ts_hat_val in tqdm.tqdm(zip(ts_val_list, ts_hat_val_list))
        ])

        # Mase
        mase_errors = np.array([
            np.array(
                [self.mase(ts, ts_val, pred, frcy) for pred in ts_hat_val]
            ) for ts, ts_val, ts_hat_val \
            in tqdm.tqdm(zip(ts_list, ts_val_list, ts_hat_val_list))
        ])

        #Naive2
        #Smape of naive2
        print('Naive 2 errors')
        mean_smape_naive2 = np.array([
             self.smape(ts_val, ts_pred) for ts_val, ts_pred \
             in tqdm.tqdm(zip(ts_val_list, ts_val_naive2))
        ]).mean()

        # MASE of naive2
        mean_mase_naive2 = np.array([
             self.mase(ts, ts_val, ts_pred, frcy) \
             for ts, ts_val, ts_pred \
             in tqdm.tqdm(zip(ts_list, ts_val_list, ts_val_naive2))
        ]).mean()

        # Contribution to the owa error
        contribution_to_owa = (smape_errors/mean_smape_naive2) + \
                               (mase_errors/mean_mase_naive2)
        contribution_to_owa = contribution_to_owa/2

        return contribution_to_owa

    # Eval functions
    def smape(self, ts, ts_hat):
        # Needed condition
        assert ts.shape == ts_hat.shape, "ts must have the same size of ts_hat"

        num = np.abs(ts-ts_hat)
        den = np.abs(ts) + np.abs(ts_hat) + 1e-5
        return 2*np.mean(num/den)

    def mase(self, ts_train, ts_test, ts_hat, frcy):
        # Needed condition
        assert ts_test.shape == ts_hat.shape, "ts must have the same size of ts_hat"

        rolled_train = np.roll(ts_train, frcy)
        diff_train = np.abs(ts_train - rolled_train)
        den = diff_train[frcy:].mean() + 1e-5 #

        return np.abs(ts_test - ts_hat).mean()/den

     # Objective function for xgb
    def fforma_objective(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> (np.ndarray, np.ndarray):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        #print(y)
        n_train = len(y)
        #print(predt.shape)
        #print(predt)
        preds_transformed = predt#softmax(predt, axis=1)#np.array([softmax(row) for row in predt])
        #https://github.com/dmlc/xgboost/blob/0ddb8a7661013c75674704872c22273415268dbd/demo/guide-python/custom_objective.py#L29-L33
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))
        #print(weighted_avg_loss_func.shape)
        grad = preds_transformed*(self.contribution_to_error[y, :] - weighted_avg_loss_func)
        #print(grad)
        #grad = preds_transformer*self.contribution_to_error[y, :]
        #print(grad.shape)
        hess = self.contribution_to_error[y,:]*preds_transformed*(1.0-preds_transformed) - grad*preds_transformed
        #hess = grad*(1 - 2*preds_transformed)
        #hess = self.contribution_to_error[y, :]
        #print(grad)
        #print(hess)
        return grad.flatten()/np.linalg.norm(grad), hess.flatten()/np.linalg.norm(grad)

    def fforma_loss(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> (str, float):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        #print(predt.shape)
        #print(predt)
        preds_transformed = predt#softmax(predt, axis=1)#np.array([softmax(row) for row in predt])
        #print(preds_transformed)
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_error[y, :]).sum(axis=1).reshape((n_train, 1))
        fforma_loss = weighted_avg_loss_func.sum()/n_train
        #print(grad)
        return 'FFORMA-loss', fforma_loss

    def softmax_objective(self, preds, dtrain):
        """Softmax objective.
        Args:
            preds: (N, K) array, N = #data, K = #classes.
            dtrain: DMatrix object with training data.

        Returns:
            grad: N*K array with gradient values.
            hess: N*K array with second-order gradient values.
        """
        # Label is a vector of class indices for each input example
        labels = dtrain.get_label()
        # When objective=softprob, preds has shape (N, K)
        labels = OneHotEncoder(sparse=False).fit_transform(labels.reshape(-1, 1))
        grad = preds - labels
        hess = 2.0 * preds * (1.0-preds)
        # Return as 1-d vectors
        return grad.flatten(), hess.flatten()



    # Functions for training xgboost
    def _train_xgboost(self, params):

        if self.custom_objective:
            gbm_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                obj=self.objective,
                num_boost_round=100,
                feval=self.loss,
                evals=[(self.dtrain, 'train'), (self.dvalid, 'eval')],
                early_stopping_rounds=10,
                verbose_eval = True
            )
        else:
            gbm_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                #obj=self.softmaxobj,
                num_boost_round=100,
                #feval=self.loss,
                evals=[(self.dtrain, 'train'), (self.dvalid, 'eval')],
                early_stopping_rounds=10,
                verbose_eval = True
            )

        return gbm_model

    def _score(self, params):

        #training model
        gbm_model = self._train_xgboost(params)

        predictions = gbm_model.predict(
            self.dvalid,
            ntree_limit=gbm_model.best_iteration + 1#,
            #output_margin = True
        )

        #print(predictions)

        loss = self.fforma_loss(predictions, self.dvalid)
        # TODO: Add the importance for the selected features
        #print("\tLoss {0}\n\n".format(loss))

        return_dict = {'loss': loss[1], 'status': STATUS_OK}

        return return_dict

    def _optimize_xgb(self, threads, random_state, max_evals):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        space = {
            'n_estimators': hp.quniform('n_estimators', 1, 250, 1),
            'eta': hp.quniform('eta', 0.001, 1, 0.05),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth':  hp.choice('max_depth', np.arange(6, 15, dtype=int)),
            #'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
            #'gamma': hp.quniform('gamma', 0.1, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
        }
        space = {**space, **self.init_params}
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(self._score, space, algo=tpe.suggest,
                    # trials=trials,
                    max_evals=max_evals)
        return best

    def _wrapper_best_xgb(self, threads, random_state, max_evals):

        # Optimizing xgbost
        best_hyperparameters = self._optimize_xgb(threads, random_state, max_evals)

        best_hyperparameters = {**best_hyperparameters, **self.init_params}

        # training optimal xgboost model
        gbm_best_model = self._train_xgboost(best_hyperparameters)

        return gbm_best_model

    def train(self, X_feats=None, y_best_model=None,
              contribution_to_error=None, n_models=None,
              max_evals=None, random_state=110, threads=None, custom_objective=None,
              bayesian_opt=False, xgb_params=None):
        """
        Train xgboost with randomized
        """
        # nthreads params
        if threads is None:
            threads = mp.cpu_count() - 1

        if X_feats is not None:
            self.X_feats = X_feats

        if y_best_model is not None:
            self.y_best_model = y_best_model

        if contribution_to_error is not None:
            self.contribution_to_error = contribution_to_error

        if n_models is not None:
            self.n_models = n_models

        if max_evals is not None:
            self.max_evals = max_evals


        # Train-validation sets for XGBoost
        self.X_train_xgb, self.X_val, self.y_train_xgb, \
            self.y_val, self.indices_train, \
            self.indices_val = train_test_split(
                self.X_feats,
                self.y_best_model,
                np.arange(self.X_feats.shape[0]),
                random_state=random_state,
                stratify = self.y_best_model#.values
            )

        self.init_params = {
            'objective': 'multi:softprob',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            'num_class': self.n_models,
            'nthread': threads,
            #'booster': 'gbtree',
            #'tree_method': 'exact',
            'silent': 1,
            'seed': random_state#,
            #'disable_default_eval_metric': 1
        }


        if custom_objective:
            if custom_objective=='fforma':
                self.dtrain = xgb.DMatrix(data=self.X_train_xgb, label=self.indices_train)
                self.dvalid = xgb.DMatrix(data=self.X_val, label=self.indices_val)
            elif custom_objective=='softmax':
                self.dtrain = xgb.DMatrix(data=self.X_train_xgb, label=self.y_train_xgb)
                self.dvalid = xgb.DMatrix(data=self.X_val, label=self.y_val)
            self.objective = self.dict_obj[custom_objective]
            self.loss = self.fforma_loss
            self.init_params['disable_default_eval_metric'] = 1
            self.custom_objective=True
        else:
            self.dtrain = xgb.DMatrix(data=self.X_train_xgb, label=self.y_train_xgb)
            self.dvalid = xgb.DMatrix(data=self.X_val, label=self.y_val)
            self.custom_objective=False

        if bayesian_opt:
            self.xgb = self._wrapper_best_xgb(threads, random_state, self.max_evals)
        else:
            # From http://htmlpreview.github.io/?https://github.com/robjhyndman/M4metalearning/blob/master/docs/M4_methodology.html
            if xgb_params:
                params = {**xgb_params, **self.init_params}
            else:
                params = {
                    'n_estimators': 94,
                    'eta': 0.01,
                    'max_depth': 14,
                    'subsample': 0.92,
                    'colsample_bytree': 0.77
                }
                params = {**params, **self.init_params}


            self.xgb = self._train_xgboost(params)

        self.opt_weights = self.xgb.predict(xgb.DMatrix(self.X_feats))#, output_margin = True)
        #self.opt_weights = softmax(self.opt_weights, axis=1)

        return self


    def predict(self, y_hat_df, y_train_df=None):
        """
        Parameters
        ----------
        y_hat_df: pandas df
            panel with columns unique_id, ds, y_{model} for each model to ensemble
        max_evals: int
        """

        y_hat_r = y_hat_df.filter(items=['unique_id', 'ds'])
        y_hat_r['y_hat'] = None

        ts_hat_list = [np.array([df[col].values for col in self.models]) for idx, df in y_hat_df.groupby('unique_id')]

        if y_train_df is None:
            ensemble = self._ensemble(ts_hat_list, self.opt_weights)

        else:
            ts_list = [df['y'].values for idx, df in y_train_df.groupby('unique_id')]

            X_feats = tsfeatures(ts_list, frcy=self.frcy)

            weights = self.xgb.predict(xgb.DMatrix(X_feats))

            ensemble = self._ensemble(ts_hat_list, weights)

        unique_ids = y_hat_df['unique_id'].unique()

        for idx, u_id in enumerate(unique_ids):
            y_hat_r.loc[y_hat_r['unique_id'] == u_id,'y_hat'] = ensemble[idx]

        return  y_hat_r


    def _ensemble(self, ts_hat_list, weights):
        """
        For each series in ts_list returns predictions
        ts_predict: list of series to predict
        """
        final_preds = np.array([np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(ts_hat_list, weights)])

        return final_preds
