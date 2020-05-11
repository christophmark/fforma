import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

class LightGBM:

    def __init__(self):
        pass

    # Functions for training lightgbm
    def _train_lightgbm(self, params):

        num_round = int(params['n_estimators'])
        del params['n_estimators']

        if self.use_cv:
            print(10*'='+'Using CV'+10*'='+'\n')
            num_round = self._train_lightgbm_cv(params, num_round)

            print('\n')
            print('Optimal rounds using CV: {}'.format(num_round))
            print('\n')

        print(10*'='+'Training FFORMA'+10*'='+'\n')
        if self.custom_objective:

            gbm_model = lgb.train(
                params=params,
                train_set=self.dtrain,
                fobj=self.objective,
                num_boost_round=num_round,
                feval=lambda preds, train_data: [self.loss(preds, train_data),
                                                 self.mse_loss(preds, train_data)],
                valid_sets=[self.dtrain, self.dvalid],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval
            )
        else:
            gbm_model = lgb.train(
                params=params,
                train_set=self.dtrain,
                num_boost_round=num_round,
                valid_sets=[self.dtrain, self.dvalid],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval
            )


        return gbm_model

    # Functions for training lightgbm
    def _train_lightgbm_cv(self, params, num_round):

        #print(self.dtrain)
        if self.custom_objective:
            gbm_model = lgb.cv(
                params=params,
                train_set=self.dtrain,
                fobj=self.objective,
                num_boost_round=num_round,
                feval=lambda preds, train_data: [self.loss(preds, train_data),
                                                 self.mse_loss(preds, train_data)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval,
                folds=StratifiedKFold(n_splits=self.nfolds).split(self.holdout_feats_train_, self.best_models_train_)
            )
        else:
            gbm_model = lgb.cv(
                params=params,
                train_set=self.dtrain,
                num_boost_round=num_round,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval = self.verbose_eval,
                folds=StratifiedKFold(n_splits=self.nfolds).split(self.holdout_feats_train_, self.best_models_train_)
            )

        optimal_rounds = len(gbm_model[list(gbm_model.keys())[0]])

        return optimal_rounds


class XGBoost:

    def __init__(self):
        passed

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
