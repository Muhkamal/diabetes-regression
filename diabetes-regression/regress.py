"""Predict diabetes disease progression after one year (442 patients).

Real public benchmark data (Efron et al., bundled with scikit-learn).
Compares linear models and tree ensembles against a predict-the-mean baseline.
"""

import json

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

SEED = 42
data = load_diabetes()
X, y, names = data.data, data.target, list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 2, 30)),
    "Lasso": LassoCV(cv=5, random_state=SEED, max_iter=20000),
    "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED),
    "Gradient boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED
    ),
}

results = {}
preds = {}
for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    results[name] = {
        "cv_r2_mean": round(float(cv_r2.mean()), 3),
        "cv_r2_std": round(float(cv_r2.std()), 3),
        "test_r2": round(float(r2_score(y_test, p)), 3),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, p))), 2),
        "test_mae": round(float(mean_absolute_error(y_test, p)), 2),
    }

best = max((k for k in results if k != "Mean baseline"), key=lambda k: results[k]["cv_r2_mean"])
ridge = models["Ridge"]
coefs = sorted(zip(names, ridge.coef_), key=lambda t: -abs(t[1]))

out = {
    "n_samples": int(len(y)),
    "n_features": int(X.shape[1]),
    "target_mean": round(float(y.mean()), 1),
    "target_std": round(float(y.std()), 1),
    "best_model_by_cv": best,
    "results": results,
    "scatter": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(y_test, preds[best])],
    "ridge_alpha": round(float(ridge.alpha_), 3),
    "ridge_coefs": [[n, round(float(c), 1)] for n, c in coefs],
}
json.dump(out, open("results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "scatter"}, indent=2))
