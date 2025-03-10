---
layout: single
title: "Model Selection in Linear Regression"
excerpt: ""
categories:
  - Intro-Data-Science
tags:
  - regression
toc: true
toc_label: "Table of Contents"
#  toc_icon: 
# header:
#   image:
#   teaser:
---

# Var

## Forward Selection

$$
\begin{aligned}
&\text{Initialize:} \quad M_0 = \{\} \\
&\text{While} \quad \text{improvement} \quad \text{is significant:} \\
&\quad \text{For each} \quad X_i \quad \text{not in} \quad M_k: \\
&\quad \quad \text{Fit} \quad M_k + X_i \\
&\quad \quad \text{Select} \quad X_i \quad \text{with best improvement} \\
&\quad \text{Add} \quad X_i \quad \text{to} \quad M_k \\
\end{aligned}
$$

## Backward Elimination

$$
\begin{aligned}
&\text{Initialize:} \quad M_{\text{full}} \\
&\text{While} \quad \text{any variable is not significant:} \\
&\quad \text{For each} \quad X_i \quad \text{in} \quad M_k: \\
&\quad \quad \text{Fit} \quad M_k - X_i \\
&\quad \quad \text{Select} \quad X_i \quad \text{with least contribution} \\
&\quad \text{Remove} \quad X_i \quad \text{from} \quad M_k \\
\end{aligned}
$$

# Error Metrics

For linear regression, a smaller p-value and larger $R^2$ indicate a better model. For other models, different baselines could be used.

## RMSE

The sum of squared error $SSE$ is the error function used for training the model, but it can't compare performance on datasets of different sizes and
is in the units of $y^2$. **Root Mean Squared Error (RMSE)**, the square root of the sum of squared residuas, is on the scale of $y$ and can compare results of
inputs of different sizes. Regardless of $n$, lower $RMSE$ means less error. 

$$RMSE = \sqrt{MSE}=\sqrt{SSE/n} =  \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2} = \frac{1}{n} || y - \hat{y} ||_2$$

```python
def root_mean_squared_error(y_truth, y_pred):
    return np.sqrt(np.mean((y_truth - y_pred) ** 2))
```

Unlike in ANOVA, the $MSE$ does not account for the degrees of freedom $n-2$. 

## MAE

**Mean Absolute Error (MAE)**, the mean absolute value of residuals, is also on the scale of the data and provides a metric to compare linear models that is not 
the metric that the parameters were trained on. Least squares is equivalent to minimizing $RMSE$. The least squares model is not the model with smallest $MAE$.


$$MAE =\frac{1}{n} \sum_{i=1}^n | y_i - \hat{y}_i | = \frac{1}{n} || y - \hat{y} ||_1$$

```python
def mean_absolute_error(y_truth, y_pred):
    return np.mean(np.abs(y_truth - y_pred))
```