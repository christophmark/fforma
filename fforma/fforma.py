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
    def __init__(self, obj='owa',
                 max_evals=100, verbose_eval=True,
                 num_boost_round=100, early_stopping_rounds=10,
                 xgb_params=None,
                 bayesian_opt=False,
                 custom_objective='fforma',
                 threads=None, seed=None):
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
        self.max_evals = max_evals

        self.dict_obj = {'fforma': self.fforma_objective, 'softmax': self.softmax_objective}

        self.verbose_eval = verbose_eval
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

        self.xgb_params = xgb_params
        self.bayesian_opt = bayesian_opt

        self.custom_objective = custom_objective

        # nthreads params
        if threads is None:
            self.threads = mp.cpu_count() - 1

        if seed is None:
            self.seed = 260294

    def _prepare_to_train(self, ts_list, ts_val_list, ts_hat_val_list, frcy, obj='owa'):
        '''

        Returns tsfeatures of ts, best model for each time series
        and contribution to owa
        '''
        #n_models =
        if obj == 'owa':
            print('Computing contribution to owa')
            contribution_to_error = self.contribution_to_owa(
                ts_list, ts_val_list, ts_hat_val_list, frcy
            )
        elif obj=='rmsse':
            print('Computing contribution to rmsse')
            contribution_to_error = self.contribution_to_rmsse(
                ts_list, ts_val_list, ts_hat_val_list
            )

        best_models = contribution_to_error.argmin(axis=1)
        print('Computing features')
        ts_features = tsfeatures(ts_list, frcy=frcy)

        return ts_features, best_models, contribution_to_error

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

    def contribution_to_rmsse(self, ts_list, ts_val_list, ts_hat_val_list):

        print('Calculating errors')
        # Smape
        rmsse_errors = np.array([
            np.array(
                [self.rmsse(ts, ts_val, pred) for pred in ts_hat_val]
            ) for ts, ts_val, ts_hat_val \
            in tqdm.tqdm(zip(ts_list, ts_val_list, ts_hat_val_list))
        ])


        return rmsse_errors

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

    def rmsse(self, ts_train, ts_test, ts_hat):
        # Needed condition
        assert ts_test.shape == ts_hat.shape, "ts must have the same size of ts_hat"

        rolled_train = np.roll(ts_train, 1)
        diff_train = np.abs(ts_train - rolled_train)
        den = diff_train[1:].mean() #

        return np.sqrt(((ts_test - ts_hat)**2).mean()/den)

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
        return grad.flatten()/np.linalg.norm(grad), hess.flatten()/np.linalg.norm(hess)

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

        num_round = int(params['n_estimators'])
        del params['n_estimators']

        if self.custom_objective:
            gbm_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                obj=self.objective,
                num_boost_round=num_round,
                feval=self.loss,
                evals=[(self.dtrain, 'train'), (self.dvalid, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval
            )
        else:
            gbm_model = xgb.train(
                params=params,
                dtrain=self.dtrain,
                #obj=self.softmaxobj,
                num_boost_round=num_round,
                #feval=self.loss,
                evals=[(self.dtrain, 'train'), (self.dvalid, 'eval')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval
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

    def _optimize_xgb(self, threads, random_state, trials, max_evals):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 100, 1),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth':  hp.choice('max_depth', np.arange(30, 100, dtype=int)),
            #'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'lambda': hp.quniform('lambda', 0.1, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
        }
        space = {**space, **self.init_params}
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(self._score, space, algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_evals)
        return best

    def _wrapper_best_xgb(self, threads, random_state, trials, max_evals):

        # Optimizing xgbost
        best_hyperparameters = self._optimize_xgb(threads, random_state, trials, max_evals)

        best_hyperparameters = {**best_hyperparameters, **self.init_params}

        # training optimal xgboost model
        gbm_best_model = self._train_xgboost(best_hyperparameters)

        return gbm_best_model

    def _tsfeatures(self, y_train_df, y_val_df, freq):
        #TODO receive panel of freq
        complete_data = pd.concat([y_train_df, y_test_df.filter(items=['unique_id', 'ds', 'y'])])
        holdout_feats = tsfeatures(y_train_df)
        feats = tsfeatures(complete_data)

        return feats, holdout_feats


    def fit(self, y_train_df=None, y_val_df=None,
            val_periods=None,
            errors=None, holdout_feats=None,
            feats=None,
            freq=None, base_model=None):
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



        # Construir lista de series
        if feats is None:
            self.feats_, self.holdout_feats_ = self._tsfeatures(y_train_df, y_val_df, freq)
        else:
            assert holdout_feats is not None, "when passing feats you must provide holdout feats"
            self.feats_, self.holdout_feats_ = feats, holdout_feats

        if errors is None:
            pass
            #calculate contribution_to_error(y_train_df, y_val_df)
        else:
            selfl.contribution_to_error = errors.values
            self.best_models_ = self.contribution_to_error.argmin(axis=1)


        # Train-validation sets for XGBoost
        self.holdout_feats_train_, self.holdout_feats_val_, \
            self.best_models_train_, \
            self.best_models_val_, \
            self.indices_train_, \
            self.indices_val_ = train_test_split(self.holdout_feats_,
                                                 self.best_models_,
                                                 np.arange(self.holdout_feats_.shape[0]),
                                                 random_state=self.seed,
                                                 stratify = self.best_models_)

        self.init_params = {
            'objective': 'multi:softprob',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            'num_class': self.contribution_to_error.shape[1],
            'nthread': self.threads,
            #'booster': 'gbtree',
            #'tree_method': 'exact',
            'silent': 1,
            'seed': self.seed#,
            #'disable_default_eval_metric': 1
        }


        if self.custom_objective:
            if self.custom_objective=='fforma':
                self.dtrain = xgb.DMatrix(data=self.holdout_feats_train_, label=self.indices_train_)
                self.dvalid = xgb.DMatrix(data=self.holdout_feats_val_, label=self.indices_val_)
            elif self.custom_objective=='softmax':
                self.dtrain = xgb.DMatrix(data=self.X_train_xgb, label=self.y_train_xgb)
                self.dvalid = xgb.DMatrix(data=self.X_val, label=self.y_val)
            self.objective = self.dict_obj[self.custom_objective]
            self.loss = self.fforma_loss
            self.init_params['disable_default_eval_metric'] = 1
            self.custom_objective=True
        else:
            self.dtrain = xgb.DMatrix(data=self.X_train_xgb, label=self.y_train_xgb)
            self.dvalid = xgb.DMatrix(data=self.X_val, label=self.y_val)
            self.custom_objective=False

        if self.bayesian_opt:
            self.trials = Trials()
            self.xgb = self._wrapper_best_xgb(threads, random_state, self.trials, self.max_evals)
        else:
            # From http://htmlpreview.github.io/?https://github.com/robjhyndman/M4metalearning/blob/master/docs/M4_methodology.html
            if self.xgb_params:
                params = {**self.xgb_params, **self.init_params}
            else:
                params = {
                    'n_estimators': 50,
                    'eta': 0.52,
                    'max_depth': 40,
                    'subsample': 0.9,
                    'colsample_bytree': 0.77,
                    'lambda': 1
                }
                params = {**params, **self.init_params}


            self.xgb = self._train_xgboost(params)

        self.error_columns = errors.columns

        weights = self.xgb.predict(xgb.DMatrix(self.feats_))
        self.weights_ = pd.DataFrame(weights,
                                     index=self.feats_.index,
                                     columns=errors.columns)

        return self


    def predict(self, y_hat_df):
        """
        Parameters
        ----------
        y_hat_df: pandas df
            panel with columns unique_id, ds, {model} for each model to ensemble
        """
        fforma_preds = self.weights_ * y_hat_df
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = 'fforma_prediction'
        preds = pd.concat([y_hat_df, fforma_preds], axis=1)

        return preds
