---
layout: single
title: "Model Selection for Linear Regression"
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

# Train-Test Split
The **train-test split** is a method used to evaluate the performance of a machine learning model. The dataset is divided into two subsets: the training set and the testing set. The model is trained on the training set and evaluated on the testing set.


## Train-Test-Validate
The **train-test-validate** approach involves splitting the dataset into three subsets: training, validation, and testing. The model is trained on the training set, tuned on the validation set, and evaluated on the testing set. The test data is reserved until after models are selected and compared. to provide an unseen dataset to test whether the model selection process and final model is accurate on unseen data


## Cross-fold Validation
**Cross-fold validation** (also called $k$-folds or CV) is a technique where the dataset is divided into $k$ subsets. The model is trained and evaluated \( k \) times, each time using a different fold as the testing set and the remaining folds as the training set. The final performance is the average of the \( k \) evaluations.


## 

# Variable Selection

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