import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
import numpy as np
import copy
from sklearn.model_selection import train_test_split

class LightGBM:

    def __init__(self):
        pass

    # Functions for training lightgbm
    def _train_lightgbm(self, holdout_feats, best_models,
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

        print(10*'='+'Training FFORMA'+10*'='+'\n')
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

    # Functions for training lightgbm
    def _train_lightgbm_cv(self, holdout_feats, best_models,
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

            optimal_gbm_model = self._train_lightgbm(holdout_feats, best_models,
                                                     params, fobj, feval,
                                                     early_stopping_rounds,
                                                     verbose_eval, seed)

            return optimal_gbm_model

        return optimal_rounds, best_performance

    def _train_lightgbm_cv_optimal_params(self, param_grid):

        best_params = {}
        best_performance = np.inf
        for iter, params in enumerate(ParameterGrid(param_grid), start=1):

            params = {**params, **self.init_params}
            num_round, performance = self._train_lightgbm_cv(params)

            if self.verbose_eval_grid:
                if isinstance(self.verbose_eval_grid, int):
                    if iter % self.verbose_eval_grid == 0:
                        print('Searching: {}'.format(iter))
                        print('Loss CV: {}'.format(performance))
                        print('\n\n')
                else:
                    print('Searching: {}'.format(iter))
                    print('Loss CV: {}'.format(performance))

                    print('\n\n')

            if performance < best_performance:
                best_params = params
                best_performance = performance
                best_params['n_estimators'] = num_round


        return self._train_lightgbm(best_params)





class XGBoost:

    def __init__(self):
        passed

    # Functions for training xgboost
    def _train_xgboost(self, params):

        num_round = int(params['n_estimators'])
        del params['n_estimators']

        if self.objective:
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
            ntree_limit=gbm_model.best_iteration #,
            #output_margin = True
        )

        #print(predictions)

        loss = self.loss(predictions, self.dvalid)
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
            'max_depth':  hp.choice('max_depth', np.arange(10, 100, dtype=int)),
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
