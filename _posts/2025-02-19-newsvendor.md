---
layout: single
title: "Newsvendor"
excerpt: ""
categories:
  - Probabilistic Models
tags:
  - classification
toc: true
toc_label: "Table of Contents"
#  toc_icon: 
# header:
#   image:
#   teaser:
---

# The Newsvendor Problem

The newsvendor model is a fundamental stochastic models used for inventory ordering and demand management. In the newsvendor problem, the decision-make must determine the order quantity $q$ to supply during a period of demand. The initial inventory is zero and the order quantity $q$ is chosen before knowing the actual demand. The optimal order quantity, $q$\*, is the one that maximizes profit or minimizes cost.

Several key variables define the problem:
- $ D $: random variable for demand with known distribution
- $ p $: unit sales price, the revenue generated per unit demanded
- $ c_p $: unit purchase or production cost, the cost incurred per unit supplied
- $ c_h $: unit holding cost, which applies when there is leftover inventory at the end of the period.
- $ c_b $: unit backorder cost, which applies when demand in the period exceeds supply and leads to replacement orders, compensation to customers, or loss of goodwill

## Objective

The objective of the newsvendor problem is to determine $ q $ such that expected profit is maximized. How can we write a function for the profit? Profit equals revenue minus cost. Cost includes purchase/production cost, holding cost, and backorder cost. Revenue is the sale price $p$ times the number of units sold. Purchase/production cost is the unit cost $c_p$ times the quantity $q$. Similarly, holding cost is $c_h \cdot \text{#units overstocked}$ and backorder cost is $c_h \cdot \text{#units understocked}$. If the demand is larger than supply, then $q$ units are sold; otherwise $D\leq q$ and $D$ units are sold. Thus, $\text{#units sold}=\min(D,q)$. The number of units overstock is how many units are supplied and not demanded. If $D>q$ then $\text{#units overstocked}=q-D$; otherwise $D\leq q$ and $\text{#units overstocked}=0$. These two cases can be written in a single expression as $\text{#units overstocked}= \max(0,q-D)$. Number of units understocked can be derived in a similar fashion. $\text{#units understocked}$ equals $q-D$ when $D>q$ and equals zero otherwise; thus  $\text{#units understocked}= \max(0, D-q)$. Putting the expressions together, the profit is

$$
\Pi(D, q) = P \cdot \min(D, q) - c_p \cdot q - c_h \cdot \max(0, q - D) - c_B \cdot \max(0, D - q)
$$

Since this function includes three random terms, it is more convenient to minimize a cost function that has the same solution as maximizing profit. Instead of accounting for revenue and backorder cost separately, they can be combined by considering lost revenue as a positive cost. In other words, combine the unit price $p$ and unit backorder cost $c_b$ into a unit shortage cost $c_s=p+c_b$. Then the cost function is

$$
C(D, q) = c_p q + c_h \max(0, q - D) + c_s \max(0, D - q)
$$

The two objectives have the same solution $q$\* as

$$
\Pi(D, q) = p D - C(D,q) \\
\arg\max_q \mathbb{E}[\Pi(D, q)] = \arg\min_q \mathbb{E}[C(D, q)]
$$

Profit and cost are random variables and the objective is to maximize/minimize their expected value. In more advanced probabilistic models, characteristics besides expected value could be considered such as variance or probability of negative profit.

## Solution

$\mathbb{E}[C(D, q)]$ is a convex function with respect to $q$ therefore $q$\* is the global minimum. $\mathbb{E}[C(D, q)]$ is decreasing before $q$\* and increasing after $q$\* so we can find $q$ from the first order condition. In other words, $q$^* is the first $q$ with nondecreasing cost.

$$q^* =\min\{q : \mathbb{E}[C(D,q+1) - C(D,q)] \geq 0\}$$

Suppose demand and quantity are integer value and $D\sim \text{ pmf } f(x) = \mathbb{P}(D=x)$. Then we can derive the solution.

$$
\begin{aligned}
& C(D, q)=c_p q+c_h \max (0, q-D)+c_s \max (0, p-q) \\
& E[C(D, q)]=c_p q+c_h \sum_{x=0}^q(q-x) f(x)+c_s \sum_{x=q+1}^{\infty}(x-q) f(x) \\
& E[C(D, q+1)]-E[C(D, q)]= c_p(q+1-q) +c_h\left(\sum_{x=0}^{q+1}(q+1-x) f(x)-\sum_{x=0}^q(q-x) f(x)\right) \\
& \qquad\qquad +c_s\left(\sum_{x=q+2}^{\infty}(x-(q+1)) f(x)-\sum_{x=q+1}^{\infty}(x-q) f(x)\right) \\
&  E[C(D, q+1)]-E[C(D, q)] =c_p+c_h \sum_{x=0}^q f(x)-c_s \sum_{x=q+1}^{\infty} f(x) \\
&  E[C(D, q+1)]-E[C(D, q)] =c_p+c_h P(D \leq q)-c_s(1-P(D \leq q)) \\
\end{aligned}
$$

Then,
$$ q^* = \min q : c_p+c_h P(D \leq q)-c_s(1-P(D \leq q)) \geq 0 $$
$$ q^* = \min q : P(D \leq q) \geq \frac{c_s-c_p}{c_s+c_h} $$

$\frac{c_s-c_p}{c_s+c_h}$ is the critical ratio and $P(D \leq x)$ is the cumulative distribution function (CDF) $F(x)$. The optimality condition could also be written with $F$ or its inverse.
$$F(q) \geq \frac{c_s-c_p}{c_s+c_h} $$
$$q \geq F^{-1}\bigg(\frac{c_s-c_p}{c_s+c_h}\bigg) $$

## Example

You run an food truck that sells burgers in a football stadium. The ingredients for a burger are purchased from a grocery store at 4 dollars per burger and your employee makes burgers quickly for a labor cost of 1 dollar per burger. Each burger sells for 10 dollars during the game and after the game, you can easily sell all the remaining cooked burgers for 3 dollars. If you run out of burgers, you will give each remaining customer a free soda as an apology for the cost of 1 dollar.

The quantity of burgers requested during a game is assumed to be between 20 and 30 with equal probability for each quantity.

$$ \frac{c_s-c_p}{c_s+c_h}=\frac{11-5}{11-3}=\frac{6}{8}=\frac{3}{4}=0.75 $$

$$D \sim Uniform(20,30) $$

$$ f(x)=\left\{\begin{array}{cc}
\frac{1}{30-20+1} = \frac{1}{11} & x \in\{20,21, \ldots, 30\} \\
0 & \text { ow }
\end{array}\right. $$

$$ F(x)=\left\{\begin{array}{cc}
0 & x<20 \\
\frac{\lfloor x\rfloor-20+1}{11} & 20\leq x<30 \\
1 & 30 \leq x
\end{array}\right.$$

$$F(27)=\frac{8}{11}= 0.727272... \qquad F(28)=\frac{9}{11}= 0.818181...$$

$$\therefore q^*= 28 $$

# Newsvendor with Inventory on Hand

$$ E[C(D, s)]=E[C(D, S)]+c_f $$
$$ c_p s+c_h \sum_{x=0}^s(s-x) f(x)+c_s \sum_{x=s+1}^{\infty}(x-s) f(x) = $$
$$ c_p S+c_h \sum_{x=0}^S (S-x) f(x)+c_s \sum_{x=0}^{\infty}(x-S) f(x)+c_f $$

$ S := q^* $

## Example


## Continuous D?

$\mathbb{E}[\max(0,q-D)] = \int_0^q(q-x) f(x) dx$

$\mathbb{E}[\max(0,D-q)] = \int_q^{\infty}(x-q) f(x) dx$

$$F(q^*)=\frac{c_s-c_p}{c_s+c_h}$$