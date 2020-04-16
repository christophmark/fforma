import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, train_test_split
from tsfeatures import tsfeatures

def mean_squared_differences(y):
    return y.diff().pow(2).mean()

class PanelModel:

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        assert X.index.names == ['unique_id', 'ds']
        self.model_ = {}
        for uid, X_group in X.groupby('unique_id'): 
            y_group = y.loc[uid]
            self.model_[uid] = clone(self.model)
            self.model_[uid].fit(X_group.values, y_group.values)
        return self

    def predict(self, X):
        idxs, preds = [], []
        for uid, X_group in X.groupby('unique_id'):
            yp = self.model_[uid].predict(X_group.values)
            idxs.extend(X_group.index)
            preds.extend(yp)
        idx = pd.MultiIndex.from_tuples(idxs, names=('unique_id', 'ds'))
        preds = pd.Series(preds, index=idx)
        return preds

class FForma(BaseEstimator, RegressorMixin):

    def __init__(self, models, cv, freq):
        self.models = models
        self.cv = cv
        self.freq = freq

    def fit(self, X, y):
        # Obtener MSE para cada serie y modelo usando CV
        errors = {}
        ts_list, uids = [], []
        for uid, X_group in X.groupby('unique_id'):
            y_group = y.loc[uid].values
            uids.append(uid)
            ts_list.append(y_group)
            group_errors = []

            for model_name, model in self.models.items():
                scores = cross_val_score(model, 
                                         X_group.values, y_group, 
                                         cv=self.cv,
                                         scoring='neg_mean_squared_error')
                group_errors.append(-scores.mean()) 
                
            errors[uid] = group_errors

        # Ajustar modelos base
        self.models_ = {}
        for model_name, model in self.models.items():
            panel_model = PanelModel(model)
            panel_model.fit(X, y)
            self.models_[model_name] = panel_model
        
        # Obtener RMSSE de cada serie y modelo
        errors = pd.DataFrame.from_dict(errors, orient='index', 
                                        columns=self.models.keys())
        denoms = y.groupby('unique_id').apply(mean_squared_differences)
        errors = errors.div(denoms, axis=0)
        errors = np.sqrt(errors)

        # Entenar xgboost
        xgb_X = tsfeatures(ts_list, self.freq) 
        xgb_y = errors.values.argmin(axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(xgb_X, xgb_y, 
                                                              stratify=xgb_y)
        params = {
                'objective': 'multi:softprob',
                'num_class': len(self.models),
                'nthread': None,
                'silent': 1,
                'seed': 0,
        }

        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        
        xgb_model = xgb.train(params=params, dtrain=dtrain, 
                              evals=[(dtrain, 'train'), (dvalid, 'eval')])

        # Obtener pesos para el ensamble
        weights = xgb_model.predict(xgb.DMatrix(xgb_X))
        self.weights_ = pd.DataFrame(weights, 
                                     index=pd.Index(uids, name='unique_id'),
                                     columns=errors.columns)
         
        return self

    def predict(self, X):
        # Obtener predicciones de los modelos base
        base_preds = pd.DataFrame({model_name: model.predict(X) 
                                  for model_name, model in self.models_.items()})

        # Ponderar con los pesos obtenidos
        fforma_preds = self.weights_ * base_preds
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = 'fforma_prediction'
        preds = pd.concat([base_preds, fforma_preds], axis=1)
        return preds

