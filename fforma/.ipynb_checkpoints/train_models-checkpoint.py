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

    def train(self, basic_models, ts_list, frcy):
        """
        basic_models: List of models
        """
        self.fitted_models = [
            np.array([self.train_basic(model, ts, frcy) for model in basic_models]) for ts in tqdm.tqdm(ts_list)
        ]

        return self

    def predict(self, h):
        if not isinstance(h, list):
            y_hat = [
                np.array([model.predict(h) for model in idts]) for idts in tqdm.tqdm(self.fitted_models)
            ]
        else:
            y_hat = [
                np.array([model.predict(h[idh]) for model in idts]) for idh, idts in tqdm.tqdm(enumerate(self.fitted_models))
            ]

        return y_hat