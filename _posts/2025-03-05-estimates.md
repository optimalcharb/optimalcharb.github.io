---
layout: single
title: "Statistical Estimation"
excerpt: "How to estimate a parameter of a probability distribution"
categories:
  - Intro-Data-Science
tags:
  - estimation
toc: true
toc_label: "Table of Contents"
#  toc_icon: 
# header:
#   image:
#   teaser:
---

# What is an estimate?

An **estimate** (or estimator) predicts a parameter from sample data. This post discusses point estimates, estimates that predict a single value.
Interval estimates (or confidence intervals) provide a range of values that contain the parameter with a set level of confidence.

## Bias

The **bias** of an estimate is the expected difference between the estimate and true parameter value. An estimator is **unbiased** if the bias is zero.

$$Bias(\hat{\theta}) = \mathbb{E}[\hat{\theta} - \theta] = \mathbb{E}[\hat{\theta}] - \theta
$$

## Example

$X_1, X_2, ..., X_n$ are i.i.d. (independent and identically distributed) random variables with mean $\mu$. Given a sample dataset $x_1, ..., x_n$, one estimate for $\mu$ is the sample mean.

$$\hat{\mu}_1 = \frac{1}{n} \sum_{i=1}^n x_i
$$

Another estimate is a time-weighted average of the samples.

$$\hat{\mu}_2 = \frac{1\cdot x_1 + 2\cdot x_2 + ... + n \cdot x_n}{1+2+...+n} = \frac{\sum_{i=1}^n i \cdot x_i}{\sum_{i=1}^n i} = \frac{\sum_{i=1}^n i \cdot x_i}{n(n+1)/2}
$$

Both are unbiased estimates.

$$Bias(\hat{\mu}_1) = \mathbb{E}[\frac{1}{n} \sum_{i=1}^n x_i] - \mu = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[x_i] - \mu = \frac{1}{n} n \mu - \mu = 0
$$
$$\begin{aligned}
Bias(\hat{\mu}_2) & = \mathbb{E}[\frac{\sum_{i=1}^n i \cdot x_i}{\sum_{i=1}^n i}] - \mu = \frac{\sum_{i=1}^n i \cdot \mathbb{E}[x_i]}{\sum_{i=1}^n i} - \mu
 \\
& = \frac{\sum_{i=1}^n i \cdot \mu}{\sum_{i=1}^n i}  - \mu = \frac{\mu \sum_{i=1}^n i}{\sum_{i=1}^n i}  - \mu = \mu - \mu = 0
\end{aligned}
$$


# Method of Moments Estimation

Method of moments is a simple approach to get an intuitive estimate of the parameters.
It uses the $k$-th population moment calculated from the pdf and $k$-th sample moment calculated from the sample.
The $k$-th population moment of a random variable $X$ is $$\mathbb{E}[X^k]$$. The $k$-th sample moment is the mean observed value in a sample $$\frac{1}{n}\sum_{i=1}^{n} x_i^k$$.

The method of moment estimates the parameters as the values that result in population moments equal to the sample moments. To estimate $k$ parameters, the first $k$ moments are used.

$$
\begin{cases}
\mathbb{E}[X] = \frac{1}{n} \sum_{i=1}^{n} x_i \\
\mathbb{E}[X^2] = \frac{1}{n} \sum_{i=1}^{n} x_i^2 \\
\vdots \\
\mathbb{E}[X^k] = \frac{1}{n} \sum_{i=1}^{n} x_i^k
\end{cases}
$$

Shorthand notation for the $k$-th sample moment is $$\bar{X^k}$$.
Method of moments depends on expressing the population moments as a function of the parameters, which may be impossible if the pdf has no closed form integral.

## Exponential

To estimate a single parameter, only the first moment (the mean) is needed.
For example, the method of moments estimate for $\lambda$ when $$X\sim \text{Exponential}(\lambda)$$ uses the mean $$\mathbb{E}[X]=1/\lambda$$ to produce the estimate:

$$\frac{1}{\hat{\lambda}}=\frac{1}{n} \sum_{i=1}^{n} x_i $$.

$$\hat{\lambda}=\frac{n}{\sum_{i=1}^{n} x_i} $$.

## Normal
Method of moments can quickly estimate the parameters $$\mu, \sigma^2$$ of the Normal distribution.
$\mu$ is the mean of the distribution for a random variable $X$, $$\mathbb{E}[X]$$, and $$\sigma^2$$ is the variance of the distribution $\text{Var}(X)$.
These parameters could still be estimated for a non-normal random variable. Then they would still represent the mean and variance, but not appear in the pmf/pdf.

Recall the computation of variance

$$\text{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2$$

Then the second moment $$\mathbb{E}[X^2]$$ can be written in terms of the parameters

$$\mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2 = \sigma^2 + \mu^2$$

The method of moment system is

$$
\begin{cases}
\hat{\mu}  = \frac{1}{n} \sum_{i=1}^{n} x_i \\
\hat{\sigma}^2 + \hat{\mu}^2 = \frac{1}{n} \sum_{i=1}^{n} x_i^2
\end{cases}
$$

In the first equation, the true population mean $\mu$ is estimated as the sample mean $\hat{\mu}=\bar{X}$. Substituing the first equation into the second provides

$$\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} x_i^2 - (\frac{1}{n} \sum_{i=1}^{n} x_i) ^ 2 $$

This means that the method of moments estimate for $\sigma^2$ is an unbiased version of the sample variance since

$$ \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{X})^2 $$

The sample variance $$S^2$$ is not an unbiased estimate for the true population variance. The sample variance instead subtracts one from the denominator.

$$ S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2 $$

$$ \mathbb{E}[S^2] = \frac{n}{n-1} \sigma^2 $$

Although the bias decreases with $n$ so for large $n$ the sample variance is close to unbiased.

$$Bias(S^2) = \mathbb{E}[S^2 - \sigma^2] = (\frac{n}{n-1} - 1) \sigma^2  = \frac{\sigma^2}{n-1} $$

# Maximum Likelihood Estimation
**Maximum Likelihood Estimation (MLE)** is a [data model]({% link _posts/2025-02-20-intro.md %}) that chooses the parameter estimates that maximize the likelihood of the data sample given the parameters.

$$\hat{\theta} = \arg \max_\theta L(\theta)$$

The likelihood function $L(\theta)$ represents the probability of observing the sample data given that $\theta$ is the true parameter value.
Let $f(x; \theta)$ be the pmf/pdf of a random variable $X$ given parameter value $\theta$ and a dataset $x_1, x_2, ..., x_n$. Then the likelihood is

$$L(\theta; x_1, ..., x_n) = \prod_{i=1}^{n} f(x_i; \theta) $$

To find the maximum likelihood solution $\hat{\theta}$, it is easier to maximize the loglikelood function 

$$l(\theta) = \log L(\theta) = \log \prod_{i=1}^{n} f(x_i; \theta) = \sum_{i=1}^{n} \log(f(x_i; \theta)) $$

The input that maximizes the loglikelihood must maximize the likelihood since log doesn't change the position of the max.

$$\hat{\theta} = \arg \max_\theta L(\theta) = \arg \max_\theta l(\theta)$$

The loglikelihood function is often concave so the maximum can be found by taking the first derivative (gradient) and setting it equal to zero.

## Exponential
For example, let's find the MLE of the Exponential distribution. $X\sim \text{Exponential}(\lambda)$ has pdf

$$ f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0 $$

The loglikelihood function and derivative follow.

$$l(\lambda) = \sum_{i=1}^{n} \log(\lambda e^{-\lambda x}) = n \log \lambda - \lambda \sum_{i=1}^n x_i$$
$$ \frac{dl}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} x_i $$

The MLE has derivative equal to zero
$$\frac{n}{\lambda} - \sum_{i=1}^{n} x_i = 0 $$
$$ \hat{\lambda} = \frac{n}{\sum_{i=1}^{n} x_i} $$

## Normal

For a normal distribution with mean $\mu$ and variance $\sigma^2$, the pdf is
$$ f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $$

The likelihood is 
$$ L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu)^2}{2\sigma^2}} $$

The loglikelihood is
$$ l(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2 $$

To find the MLEs, we take the partial derivatives of $l$ with respect to $\mu$ and $\sigma^2$ and set them equal to zero. For $\mu$

$$ \frac{\partial l}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i - \mu) = 0 $$
$$ \hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

For $\sigma^2$

$$ \frac{\partial l}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^{n} (x_i - \mu)^2 = 0 $$
$$ \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2 $$

For Exponential and Normal, the maximum likelihood estimates are the same as the method of moments estimates.
That's a good illustration of why method of moments is a good rule of thumb for simple distributions.
However, MLE is generally a better method and finds the optimal solution for more distributions.
Even when the likelihood is not concave, the first order approximation of a local maximum is often good enough.

# Estimate Evaluation
How should an estimate be selected? MLE produces good estimates, but they can be biased or higher variance than other estimates. The "best" estimate could be considered to be the estimate with lowest MSE or the most efficient estimate. Even when estimates besides MLE are not considered, the MSE, efficiency, and sufficiency of the MLE describe its quality.

## Mean Squared Error (MSE)
In addition to bias, mean squared error is a metric to evaluate estimates. **Mean squared error (MSE)** is the mean of the squared difference between the estimate and true parameter value.

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] \qquad \text{(definition)}
$$

The definition can be simplified to use in computations.

$$
\begin{aligned}
\text{MSE}(\hat{\theta})
& = \mathbb{E}[(\hat{\theta} - \theta)^2] \\
& = \mathbb{E}[(\hat{\theta} - \mathbb{E}\hat{\theta} + \mathbb{E}\hat{\theta} - \theta)^2] \\
& = \mathbb{E}[(\hat{\theta} - \mathbb{E}\hat{\theta})^2 + 2(\hat{\theta} - \mathbb{E}\hat{\theta})(\mathbb{E}\hat{\theta} - \theta) + (\mathbb{E}\hat{\theta} - \theta)^2] \\
& = \mathbb{E}[(\hat{\theta} - \mathbb{E}\hat{\theta})^2 + 2(0)(\mathbb{E}\hat{\theta} - \theta) + (\mathbb{E}\hat{\theta} - \theta)^2] \\
& = \mathbb{E}[(\hat{\theta} - \mathbb{E}\hat{\theta})^2] + (\mathbb{E}\hat{\theta} - \theta)^2 \\
\text{MSE}(\hat{\theta}) & = Var(\hat{\theta}) + Bias(\hat{\theta})^2 \quad \text{(computation)}
\end{aligned}
$$

Thus, MSE is a metric that considers both variance and bias. MSE could be used to compare estimates. However, when making predictions, lack of bias is preferable to lower variance. So the best estimate is an unbiased estimate with the lowest MSE and thus lowest variance. This is called an miminum variance unbiased estimate or **efficient estimate**.

## Efficient Estimate and Cramer-Rao Bound

An efficient estimate acheives the minimum variance among all unbiased estimators.
The minimum possible variance is not a straightforward computation from a model, but a lower bound is known.
The **Cramer-Rao Bound** provides a lower bound on the variance of any unbiased estimator $\hat{\theta}$ of a parameter $\theta$.
An estimate can be compared against this bound to determine if the variance is relatively small and the estimate relatively good.
The Cramer-Rao Bound is

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

where $\(I(\theta)\)$ is the Fisher Information, which is the expected squared derivative of the loglikelihood (see [MLE](#maximum-likelihood-estimation) ) of the data $X$ given $\theta$

$$
I(\theta) = \mathbb{E} \left[ \left( \frac{\partial}{\partial \theta} \log L(X; \theta) \right)^2 \right]
$$

The Fisher Information quantifies the amount of information that an observable random variable $X$ provides about an unknown parameter $\theta$.

## Sufficient Estimate and Fisher-Neyman Theorem

Let $X=(X_1,X_2,...,X_N)$ be a vector of i.i.d. random variables, $T(X)$ be a transformation of $X$, and $x$ denote an outcome/value of $X$. Then $X_i$ has pmf/pdf $f(x_i;\theta)$ with parameter $\theta$ and $X$ has joint pmf/pdf $f(x;\theta) $ $=$ $f(x_1,x_2,...,x_n; \theta) $ $ =$ $\prod_{i=1}^n f(x_i; \theta)$. 
An estimate for $\theta$ calculated from a data sample is not necessarily a sufficient statistic. However, any efficient estimate for $\theta$ must be a function of a sufficient statistic $T(X)$. Definition: $T(X)$ is a sufficient statistic if the probability distribution of $X$ given a value for $T(X)$ is constant with respect to $\theta$. In math notation, $f(x \| T(X)=T(x))$ is not a function of $\theta$. In other words, the statistic $T(X)$ provides as much information about $\theta$ as the entire data sample $X$.

The Fisher-Neyman Factorization Theorem provides a shortcut to prove that a statistic is sufficient. The theorem states $T(X)$ is a sufficient statistic for $\theta$ if joint pmf/pdf $f(x; \theta)$ $=$ $g(T(x), \theta) h(x)$ for some $g$ that is a function of $T(X)$ and the parameters and some $h$ that is any function of $X$ and the parameters besides $\theta$. 

For example, let $X_1, X_2, ..., X_n \sim \text{Poisson} (\lambda)$. The maximum likelihood estimate for $\lambda$ is the sample mean $\frac{1}{n} \sum_{i=1}^n x_i$.
This is a function of $T(x)$ $ =$ $\sum_{i=1}^n x_i$, which is a sufficient statistic for $\lambda$. The sufficiency can be shown by the definition or theorem.

Recall $X_i \sim \text{Poisson} (\lambda)$ has pmf
$$
f\left(x_i\right)=\frac{e^{-\lambda} \lambda^{x_i}}{x_{i}!}
$$

and $T(X)=\sum_{i=1}^n X_i$ follows a $\text{Poisson}(n\lambda)$ distribution.

To prove sufficiency using the definition, show that $f(x \| T(X)=s)$ is equivalent to a function without $\lambda$. I'm using $s$ since the statistic is the sample sum.

$$f\big(x\big| \sum_{i=1}^nx_i = s\big) = \frac{\prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_{i}!}}{\frac{e^{-n \lambda} \cdot (n \lambda)^{s} }{s!}}
$$

$$= \frac{\frac{e^{-n\lambda} \cdot \lambda^{\sum_{i=1}^n x_i}}{\prod_{i=1}^n x_{i}!}}{\frac{e^{-n \lambda} \cdot (n \lambda)^{s} }{s!}}
$$

$$= \frac{\frac{e^{-n\lambda} \cdot \lambda^{s}}{\prod_{i=1}^n x_{i}!}}{\frac{e^{-n \lambda} \cdot n^s \lambda^{s} }{s!}}
$$

$$= \frac{s! \cdot n^{-s}}{\prod_{i=1}^n x_{i}!}
$$

A proof using the theorem is much faster.

$$f(x;\lambda) = \prod_{i=1}^n f(x_i) = \prod_{i=1}^n \frac{e^{-\lambda} \lambda^{x_i} }{x_{i}!}
 = \frac{e^{-n\lambda} \lambda^{\sum_{i=1}^n x_i}}{\prod_{i=1}^n x_{i}!} $$

The numerator is a function $g$ of $T(x) = \sum_{i=1}^n x_i$ and $\lambda$ and the denominator is a function $h$ of $x$ without $\lambda$.

