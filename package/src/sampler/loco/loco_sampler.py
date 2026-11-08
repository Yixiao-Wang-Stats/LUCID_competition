import numpy as np
from ..sampler import Sampler

class LOCOSampler(Sampler):
    def __init__(
        self,
        predictive_model_class,
        predictive_model_kwargs=None,
        random_state=None,
    ):
        super().__init__(random_state)
        self.predictive_model_class = predictive_model_class
        if predictive_model_kwargs is None:
            predictive_model_kwargs = {}
        self.predictive_model_kwargs = predictive_model_kwargs
        self.models = {}
        self.feature_size = None

    def fit(self,X_train,feature_idx,feature_type=None,model=None):
        X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        feature_size = X_train.shape[1]
        if not 0 <= feature_idx < feature_size:
            raise ValueError("feature_idx is out of range.")
        if self.feature_size is None:
            self.feature_size = feature_size
        elif self.feature_size != feature_size:
            raise ValueError("X_train has a different number of features from previous fitted data.")
        if model is None:
            raise ValueError("model must be provided for LOCO.")
        X_train_minus_j = np.delete(X_train,feature_idx,axis=1)
        prediction_train = model.predict(X_train)
        reduced_model = self.predictive_model_class(**self.predictive_model_kwargs)
        reduced_model.fit(X_train_minus_j,prediction_train)
        self.models[feature_idx] = reduced_model
        self.is_fitted = True
        return self

    def predict_batch(self,X_eval,feature_idx):
        if not self.is_fitted:
            raise RuntimeError("Sampler must be fitted before prediction.")
        if feature_idx not in self.models:
            raise RuntimeError("The sampler has not been fitted for this feature.")
        X_eval = np.asarray(X_eval)
        if X_eval.ndim != 2:
            raise ValueError("X_eval must be a 2D array.")
        if X_eval.shape[1] != self.feature_size:
            raise ValueError("X_eval has the wrong number of features.")
        X_eval_minus_j = np.delete(X_eval,feature_idx,axis=1)
        return self.models[feature_idx].predict(X_eval_minus_j)

    def sample(self,*args,**kwargs):
        raise NotImplementedError("Sampling is not implemented for LOCO.")