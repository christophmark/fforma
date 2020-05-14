import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split

class FForma:

    def __init__(self):
        pass

    def fit(self, ts_features, errors):
        # Entenar xgboost
        xgb_y = errors.values.argmin(axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(ts_features, xgb_y,
                                                              stratify=xgb_y)
        params = {
                'objective': 'multi:softprob',
                'num_class': errors.shape[1],
                'nthread': None,
                'silent': 1,
                'seed': 0,
        }

        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

        xgb_model = xgb.train(params=params, dtrain=dtrain,
                              evals=[(dtrain, 'train'), (dvalid, 'eval')])

        # Obtener pesos para el ensamble
        weights = xgb_model.predict(xgb.DMatrix(ts_features))
        self.weights_ = pd.DataFrame(weights,
                                     index=errors.index,
                                     columns=errors.columns)

        return self

    def predict(self, base_preds):
        # Ponderar con los pesos obtenidos
        fforma_preds = self.weights_ * base_preds
        fforma_preds = fforma_preds.sum(axis=1)
        fforma_preds.name = 'FformaClassificationEnsemble'
        preds = pd.concat([base_preds, fforma_preds], axis=1)
        return preds

