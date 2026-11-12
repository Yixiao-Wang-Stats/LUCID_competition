"""Auto MPG LUCID uncertainty from DRBART conditional-model posterior draws."""

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import lucid


X, y = lucid.datasets.auto_mpg()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026,
)
model = XGBRegressor(random_state=2026)
model.fit(X_train, y_train)

feature_types = [
    "categorical" if name in {"cylinders", "model_year", "origin"}
    else "continuous"
    for name in X.columns
]
explainer = lucid.UncertaintyExplainer(
    model,
    X_train,
    sampler="drbart",
    feature_types=feature_types,
    random_state=2026,
)


explanation = explainer(X_test, features=["horsepower"])
lower, upper = explanation.credible_interval()
global_lower, global_upper = explanation.global_credible_interval()

print("Posterior mean LUCID:\n", explanation.values)
print("95% lower bounds:\n", lower)
print("95% upper bounds:\n", upper)
print("Global LUCID:", explanation.global_importance())
print("Global 95% interval:", global_lower, global_upper)
explainer.close()
