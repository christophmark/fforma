import tqdm
import numpy as np
import copy

class trainBasicModels:
    
    def __init__(self):
        pass
    # Train functions
    def train_basic(self, model, ts, frcy):
        this_model = copy.deepcopy(model)
        if 'frcy' in model.fit.__code__.co_varnames:
            fitted_model = this_model.fit(ts, frcy)
        else:
            fitted_model = this_model.fit(ts)

        return fitted_model

    def train(self, basic_models, y_train_df,  frcy):
        """
        basic_models: Dict of models with name
        """
        ts_list = [df['y'].values for idx, df in y_train_df.groupby('unique_id')]
        
        self.model_names = basic_models.keys()
        self.basic_models_list = basic_models.values()
        
        self.fitted_models = [
            np.array([self.train_basic(model, ts, frcy) for model in self.basic_models_list]) for ts in tqdm.tqdm(ts_list)
        ]

        return self

    def predict(self, y_hat_df):
        
        y_hat_r = y_hat_df.filter(items=['unique_id', 'ds'])
        
        h = y_hat_r.groupby('unique_id').size()[0]
        
        y_hat = [
            np.array([model.predict(h) for model in idts]) for idts in tqdm.tqdm(self.fitted_models)
        ]
        
        unique_ids = y_hat_df['unique_id'].unique()
        
        for idx, u_id in enumerate(unique_ids):
            for id_model, model in enumerate(self.model_names):
                y_hat_r.loc[y_hat_r['unique_id'] == u_id, f'y_{model}'] = y_hat[idx][id_model] 
        
        
        return y_hat_r