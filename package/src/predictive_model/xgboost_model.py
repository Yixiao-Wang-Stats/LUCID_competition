import numpy as np
import xgboost as xgb

from .predictive_model import PredictiveModel

class XGBoostModel(PredictiveModel):
    def __init__(
        self,
        objective="reg:squarederror",
        random_state=None,
        **kwargs,
    ):
        super().__init__()

        self.objective = objective
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = xgb.XGBRegressor(
            objective=objective,
            random_state=random_state,
            **kwargs,
        )


    def fit(
        self,
        X_train,
        y_train,
    ):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.model.fit(
            X_train,
            y_train,
        )

        self.is_fitted = True

        return self


    def predict(
        self,
        X,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X = np.asarray(X)

        return self.model.predict(X)


    def save(
        self,
        path,
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before saving."
            )

        self.model.save_model(path)


    @classmethod
    def load(
        cls,
        path,
    ):
        model = cls()

        model.model.load_model(path)
        model.is_fitted = True

        return model