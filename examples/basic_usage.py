"""Feature importance with Bike Sharing and TabPFN 3.5-Fast sampling."""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import lucid


X, y = lucid.datasets.bikesharing()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2026,
)
model = XGBRegressor(random_state=2026)
model.fit(X_train, y_train)
X_eval = X_test

feature_types = [
    "categorical" if name in {"hr", "season"}
    else "binary" if name == "workingday"
    else "continuous"
    for name in X.columns
]

explainer = lucid.Explainer(
    model,
    X_train,
    sampler="tabpfn",
    feature_types=feature_types,
    random_state=2026,
    n_jobs=2,
    sampler_kwargs={"version": "v3.5-fast"},
)

explanation = explainer(X_eval, n_resamples=1000)
print("LUCID values:\n", explanation.values)
print("Global LUCID (CPFI):", explanation.global_importance())

lucid.plots.bar(explanation)
lucid.plots.beeswarm(
    explanation, X_eval, categorical=["hr", "workingday", "season"],
    cmap="viridis",
)
plt.show()
explainer.close()
