# Diabetes Progression Model

Predict a disease progression score one year after baseline for 442 patients.

- **Data:** Efron et al. diabetes dataset (age, sex, BMI, blood pressure, six blood serum measures). A public benchmark bundled with scikit-learn.
- **Models:** mean baseline, linear regression, ridge, lasso, random forest, gradient boosting.
- **Method:** 80/20 split, 5-fold cross-validation on the training part, test set scored once.

## Results
| Model | CV R² | Test R² | Test RMSE |
|---|---|---|---|
| Mean baseline | -0.027 | -0.012 | 73.22 |
| Linear regression | 0.480 | 0.453 | 53.85 |
| Ridge (chosen) | 0.482 | 0.455 | 53.76 |
| Lasso | 0.480 | 0.471 | 52.92 |
| Random forest | 0.432 | 0.459 | 53.52 |
| Gradient boosting | 0.430 | 0.474 | 52.79 |

Every model beats the baseline by cutting error about 27%, and all the models land close together. The limit is the information in ten measurements, not the algorithm.

## Limits
- About half of the variation is unexplained, so individual predictions are not reliable.
- s1 and s2 (blood serum measures) are strongly correlated, so their individual coefficients should not be interpreted separately.

## Run
```bash
pip install scikit-learn numpy
python regress.py
```
