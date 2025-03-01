---
layout: single
title: "Newsvendor Problem"
excerpt: "Fundamental probabilistic model for inventory and demand"
categories:
  - Probabilistic-Models
tags:
  - stochastic
toc: true
toc_label: "Table of Contents"
#  toc_icon: 
# header:
#   image:
#   teaser:
---

# The Newsvendor Problem

The newsvendor model is a fundamental stochastic model used for inventory ordering and demand management. In the newsvendor problem, the decision-make must determine the order quantity $q$ to supply during a period of demand. The initial inventory is zero and the order quantity $q$ is chosen before knowing the actual demand. The optimal order quantity, $q$\*, is the one that maximizes profit or minimizes cost.

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

$\mathbb{E}[C(D, q)]$ is a convex function with respect to $q$ therefore $q$\* is the global minimum. $\mathbb{E}[C(D, q)]$ is decreasing before $q$\* and increasing after $q$\* so we can find $q$ from the first order condition. In other words, $q$\* is the first $q$ with nondecreasing cost.

$$q^* =\min\{q : \mathbb{E}[C(D,q+1) - C(D,q)] \geq 0\}$$

Suppose demand and quantity are integer value and $D\sim \text{ pmf } f(x) = \mathbb{P}(D=x)$. Then we can derive the solution.

$$
\begin{aligned}
& C(D, q)=c_p q+c_h \max (0, q-D)+c_s \max (0, p-q) \\
& \mathbb{E}[C(D, q)]=c_p q+c_h \sum_{x=0}^q(q-x) f(x)+c_s \sum_{x=q+1}^{\infty}(x-q) f(x) \\
& \mathbb{E}[C(D, q+1)]-\mathbb{E}[C(D, q)]= c_p(q+1-q) +c_h\left(\sum_{x=0}^{q+1}(q+1-x) f(x)-\sum_{x=0}^q(q-x) f(x)\right) \\
& \qquad\qquad +c_s\left(\sum_{x=q+2}^{\infty}(x-(q+1)) f(x)-\sum_{x=q+1}^{\infty}(x-q) f(x)\right) \\
&  \mathbb{E}[C(D, q+1)]-\mathbb{E}[C(D, q)] =c_p+c_h \sum_{x=0}^q f(x)-c_s \sum_{x=q+1}^{\infty} f(x) \\
&  \mathbb{E}[C(D, q+1)]-\mathbb{E}[C(D, q)] =c_p+c_h P(D \leq q)-c_s(1-P(D \leq q)) \\
\end{aligned}
$$

Then,
$$ q^* = \min q : c_p+c_h P(D \leq q)-c_s(1-P(D \leq q)) \geq 0 $$
$$ q^* = \min q : P(D \leq q) \geq \frac{c_s-c_p}{c_s+c_h} $$

$\frac{c_s-c_p}{c_s+c_h}$ is the critical ratio and $P(D \leq x)$ is the cumulative distribution function (cdf) $F(x)$. The optimality condition could also be written with $F$ or its inverse.
$$F(q) \geq \frac{c_s-c_p}{c_s+c_h} $$
$$q \geq F^{-1}\bigg(\frac{c_s-c_p}{c_s+c_h}\bigg) $$

## Continuous Demand

If the demand is not restricted an integer quantity like 5 items, then it can be treated as a continuous variable with values like 5.432 pounds. Now $f(x)$ represents the probability distribution function (pdf) of the demand distribution $D$. The equations for expected cost still hold, but now we must substitute different calculations for the expected understock and overstock.

$\mathbb{E}[\max(0,q-D)] = \int_0^q(q-x) f(x) dx$

$\mathbb{E}[\max(0,D-q)] = \int_q^{\infty}(x-q) f(x) dx$

If the quantity ordered is also continuous, like 3.42 gallons, then there is a point $q$\* on the cdf that produces the critical ratio exactly.

$$F(q^*)=\frac{c_s-c_p}{c_s+c_h}$$


## Example

You run an food truck that sells burgers in a football stadium. The ingredients for a burger are purchased from a grocery store at 4 dollars per burger and your employee makes burgers quickly for a labor cost of 1 dollar per burger. Each burger sells for 10 dollars during the game and after the game, you can easily sell all the remaining cooked burgers for 3 dollars. If you run out of burgers, you will give each remaining customer a free soda as an apology for the cost of 1 dollar.

The quantity of burgers requested during a game is assumed to be between 20 and 30 with equal probability for each quantity.

*Solution.* Let's find the critical ratio by substituting the appropriate parameters.


$$ \frac{c_s-c_p}{c_s+c_h}=\frac{11-5}{11-3}=\frac{6}{8}=\frac{3}{4}=0.75 $$

Since $D \sim Uniform(20,30) $, the pmf and cdf are known. The quantity that produces a cdf just above the critical ratio is the solution.

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

*Extra Practice.* With $q$\* $=28$, calculate the expected profit $\mathbb{E}[\Pi(D, q$\* $=28 )]$ and the worst case profit, $\Pi(20,28)$.

# Newsvendor with Inventory on Hand

Earlier, we determined the order quantity assuming each period started with no inventory besides the order. In many scenarios, inventory unsold from the last period can be held over into the next period. Additionally, there is a new parameter: $c_f$ the fixed cost of ordering/producing a positive number of units and is not incurred when ordering/producing zero units. For example, transportation or labor. If the initial inventory is very close to the optimal order quantity $q$\*, then it may not be worth paying the fixed cost to change the inventory going into the selling period. Thus the decision is now not only the optimal inventory $q$\* but also a cutoff for initial inventory levels to decide whether to order up to $q$\* or not order. A common notation is calling this an $(s,S)$ policy. $S$ is the optimal number of units to have at the start of the period when it is worth paying the fixed cost. $S$ is the same as $q$\* from before. $s$ represents the inventory cutoff for ordering. If inventory on hand $x$ is greather than $s$, then the policy is to not order any units and enter the period with inventory $x$. If $x<s$, then the policy is to enter the period with $S$ units which implies ordering $S-x$ units.

$S$ is found from the newsvendor critical ratio. $s$ can be found by setting the expected cost when not ordering equal to the expected cost when ordering. With discrete pmf $f$,

$$ \mathbb{E}[C(D, s)]=\mathbb{E}[C(D, S)]+c_f $$
$$ c_p s+c_h \sum_{x=0}^s(s-x) f(x)+c_s \sum_{x=s+1}^{\infty}(x-s) f(x) = $$
$$ c_p S+c_h \sum_{x=0}^S (S-x) f(x)+c_s \sum_{x=0}^{\infty}(x-S) f(x)+c_f $$

## Example

Let's use the same parameters from before, $c_s=11, c_p=5, c_h=-3,$ which produces critical ratio $0.75$. Additionally, let's add $c_f=100$. Let's practice finding $q$\* again with a new demand distribution. The demand is $10$ half of the time, $15$ a third the time, and $30$ otherwise. Now we can find $S=q$\*.

$$
D=\left\{\begin{array}{lll}
10 & \text { w.p. } & \frac{1}{2} \\
15 & \text { w.p. } & \frac{1}{3} \\
30 & \text { w.p. } & \frac{1}{6}
\end{array}\right.
$$

$$
f(x)=\left\{\begin{array}{cc}
\frac{1}{2} & x=10 \\
\frac{1}{3} & x=15 \\
\frac{1}{6} & x=30 \\
0 & \text { ow }
\end{array}\right.
$$

$$
F(x)= \begin{cases}0 & x<10 \\ \frac{1}{2} & 10 \leq x<15 \\ \frac{5}{6} & 15 \leq x<30 \\ 1 & 30 \leq x\end{cases}
$$

The two cdf values closest to the critical value are $F(10) = \frac{1}{6}$ and $F(15)=\frac{5}{6}$ so $S=15$.

Now, use $ \mathbb{E}[C(D, s)]=\mathbb{E}[C(D, S=15)]+c_f $ to find $s$. 

$$ 5 \cdot s-3(s-10) \frac{1}{2}+11(30-5) \frac{1}{6} = 5 \cdot 15-3 \cdot 5 \frac{1}{2}+11 \cdot 15 \cdot \frac{1}{6}+100 $$
$$s=\frac{115}{3}$$

Earlier we wrote expected cost and profit as a function of starting inventory and demand distribution. But under the $(s,S)$ policy, the starting inventory is not always the same. So to find the expected cost and profit, we need to find a steady-state distribution that gives the long-run probability of each starting inventory value.

# Markov Chains for $(s,S)$ policy

In addition to finding the optimal $(s,S)$ policy and finding the expected cost, we can calculate probabilities for specific changes in inventory and simulate the inventory levels across multiple periods.

Assume an optimal $(s,S)$ policy with $0 < s < S$ and the demand has known distribution $D$. Also let's consider the case when inventory and demand are integer valued. Let $X_n$ be the inventory at the start of period $n$, $Y_n$ be the inventory at the end of period $n$. If the demand in each period follows the same distribution $D$, then $ \\{X_n: n=1,2,...\\}$, $\\{Y_n: n=1,2,...\\}$ are Markov chains. 

## Sample Spaces

During the period $n$, demand is experienced and the inventory goes from $X_n$ to $Y_n$. Then inventory may be ordered or not ordered before the next period starts with $X_{n+1}$ units. Let's think about these two transitions in math notation. $Y_n = \max(0, X_n - D_n)$ where $D_n$ is the value of the random variable for demand $D$ experienced at time $n$. Note when demand is larger than starting inventory ($D>X_n$), exactly $X_n$ units are sold and $Y_n=0$. If ending inventory $Y_n < s$, then $S-Y_n$ units are ordered so that next period starts with inventory $X_{n+1}=S$. Else $Y_n \geq s$, zero units are ordered, and $X_{n+1}=Y_n$. 

What are the sample spaces (set of possible values) for $X_n$ and $Y_n$? Assuming the Markov chains have already been initialized ($Y_{n-1}$ exists and is at most $S$), the starting inventory $X_n$ will never be less than $s$ units or greater than $S$ i.e. $X_n \in \\{s, s+1, ..., S\\}$ following from the $(s,S)$ policy. Then the max value of $Y_n = \max(0, X_n - D_n)$ occurs when the starting inventory is at its maximum and demand is at its mimumum. So $X_n=S$, $D_n=\min D$, and $Y_n=\max(0,S-\min D)$. Since we assumed $0<s<S$ is optimal, $S > \min D$ and the max of $Y_n=S-\min D$. The min value of $Y_n$ occurs when $X_n=s$, $D_n=\max D$, and $Y_n= \max(0, s-\max D)$. Again using the fact $0<s<S$ is optimal, $s < \max D$ and the min of $Y_n=0$. Putting this together, $Y_n\in \\{0, 1, ..., S-\min D\\}$. The sample spaces of $X_n$ and $Y_n$ are not always these entire sets if the demand takes on a small set of values. For example, if $D$ was only even valued and $S$ was even, then the inventory would always be even.

## Example of Transition Probability Matrices

For example, let $s=3, S=7, \mathbb{P}(D=2)=0.2, \mathbb{P}(D=3)=0.5, \mathbb{P}(D=4)=0.3$. The Markov chains must also be initialized (and let's focus on the steady state where the initalized values are within the sample spaces above) $Y_0=0$ and subsequently $X_1=S$ before the first random variable $D_1$ occurs. Now we want to find $P^{(X)}$, $P^{(Y)}$, the transition probability matrices of $X_n$, $Y_n$. 

Let's think through the possible transitions of $X$ in a table. Following the logic above, $X_n$ will never be less than $s=3$ or larger than $S=7$. 
Start with the possible values for $X_n, D_n$ and then record the probability, resulting $Y_n=X_n-D_n$, order value which is $S-Y_n$ if $Y_n<s$ and $0$ if $Y_n\geq s$, and $X_{n+1}=Y_n+order$. 

| X_n  | D_n  | prob | Y_n | order | X_{n+1}  |
|----|----|------|----|-------|----|
| 3  | 2  | 0.2  | 1  | 6     | 7  |
| 3  | 3  | 0.5  | 0  | 7     | 7  |
| 3  | 4  | 0.3  | 0  | 7     | 7  |
| 4  | 2  | 0.2  | 2  | 5     | 7  |
| 4  | 3  | 0.5  | 1  | 6     | 7  |
| 4  | 4  | 0.3  | 0  | 7     | 7  |
| 5  | 2  | 0.2  | 3  | 0     | 3  |
| 5  | 3  | 0.5  | 2  | 5     | 7  |
| 5  | 4  | 0.3  | 1  | 6     | 7  |
| 6  | 2  | 0.2  | 4  | 0     | 4  |
| 6  | 3  | 0.5  | 3  | 0     | 3  |
| 6  | 4  | 0.3  | 2  | 5     | 7  |
| 7  | 2  | 0.2  | 5  | 0     | 5  |
| 7  | 3  | 0.5  | 4  | 0     | 4  |
| 7  | 4  | 0.3  | 3  | 0     | 3  |

$X_{n+1}$ never takes on the value $6$ so $6$ is actually a transient state and can be excluded from the long-run Markov chain. We could have recognized $X_n=6$ was not possible earlier because $\min D=2$ so $X_n=S=7$ can only produce values at most $5$. 

Now we can fill in the matrix $P^{(X)}$. $ P_{ij}^{(X)} $ is the probability of transitioning from $X_n=i$ to $X_{n+1}=j$ on the table. 

$$
P^{(X)} = 
\begin{array}{c|cccc}
    & 3 & 4 & 5 & 7 \\ \hline
  3 & 0   & 0   & 0   & 1   \\
  4 & 0   & 0   & 0   & 1   \\
  5 & 0.2 & 0   & 0   & 0.8 \\
  7 & 0.3 & 0.5 & 0.2 & 0   \\
\end{array}
$$

Now let's form $P^{(Y)}$ by making a table. First fill in the values 0 to 5 for $Y$ with each value repeated three time for the three possible demand values. Then find $X_{n+1}$ which is $7$ if $Y_n<3$ and $Y_n$ if $Y_n\geq 3$. Then record the possible values for $D_{n+1}$ and their probability and the resulting $Y_{n+1}=X_{n+1}-D_{n+1}$. 

| Y  | X  | D  | prob | Y  |
|----|----|----|------|----|
| 0  | 7  | 2  | 0.2  | 5  |
| 0  | 7  | 3  | 0.5  | 4  |
| 0  | 7  | 4  | 0.3  | 3  |
| 1  | 7  | 2  | 0.2  | 5  |
| 1  | 7  | 3  | 0.5  | 4  |
| 1  | 7  | 4  | 0.3  | 3  |
| 2  | 7  | 2  | 0.2  | 5  |
| 2  | 7  | 3  | 0.5  | 4  |
| 2  | 7  | 4  | 0.3  | 3  |
| 3  | 3  | 2  | 0.2  | 1  |
| 3  | 3  | 3  | 0.5  | 0  |
| 3  | 3  | 4  | 0.3  | 0  |
| 4  | 4  | 2  | 0.2  | 2  |
| 4  | 4  | 3  | 0.5  | 1  |
| 4  | 4  | 4  | 0.3  | 0  |
| 5  | 5  | 2  | 0.2  | 3  |
| 5  | 5  | 3  | 0.5  | 2  |
| 5  | 5  | 4  | 0.3  | 1  |

Now we can fill in the matrix $P^{(Y)}$. If you prefer to do thinking in your head, you can jump directly to writing down $P^{(Y)}$ instead of making a table.

$$
P^{(Y)}=
\begin{array}{c|cccccc}
    & 0 & 1 & 2 & 3 & 4 & 5 \\ \hline
  0 & 0   & 0   & 0   & 0.3 & 0.5 & 0.2 \\
  1 & 0   & 0   & 0   & 0.3 & 0.5 & 0.2 \\
  2 & 0   & 0   & 0   & 0.3 & 0.5 & 0.2 \\
  3 & 0.3 & 0.7 & 0   & 0   & 0   & 0   \\
  4 & 0.3 & 0.5 & 0.2 & 0   & 0   & 0   \\
  5 & 0   & 0.3 & 0.5 & 0.2 & 0   & 0   \\
\end{array}
$$

The Markov chain can also be visualized with a diagram:



The transition matrices can be used to calculate the steady-state distribution, probability of transitioning from $i$ to $j$ in multiple steps, expected number of steps to transition from $i$ to $j$, etc. If we find the steady-state distribution, then we can find the expected cost or profit per period. 

For a small Markov chain like $X$, we can calculate the steady-state (stationary) distribution $\pi$ by hand from $\pi P=\pi$, $\sum_i \pi_i=1$.

$$
\begin{aligned}
& \left\{\begin{array}{l}
\pi_3=0.2 \pi_5+0.3 \pi_7 \\
\pi_4=0.5 \pi_7 \\
\pi_5=0.2 \pi_7 \\
\pi_7=\pi_3+\pi_4+0.8 \pi_5 \\
\pi_3+\pi_4+\pi_5+\pi_7=1
\end{array}\right. \\
& \pi_3=0.2\left(0.2 \pi_7\right)+0.3 \pi_7=0.34 \pi_7 \\
& (0.34+0.5+0.2+1) \pi_7=1 \\
& \pi_7=\frac{1}{2.04} \approx 0.4902 \\
& \pi_3=0.34(0.4902) \approx 0.1667 \quad  \pi_4=0.5(0.4902) \approx 0.0908 \quad \pi_5 = 0.2(0.4902) \approx 0.2451
\end{aligned}
$$

For larger matrices $P$ we can find the stationary distribution by finding the positive normalized vector in $Null(P-I)$. Since $P$ is stochastic (each row sums to 1 and each value is nonnegative),  $\pi(P-I)=0$ is gauranteed to produce a solution space of dimension 1 (if $\pi \in \mathbb{R}^n$, there are $n$ equations of which any subset of $n-1$ equations is linearly independent).

## Takeaways

To minimize a function of random variables, we used the expected value of the objective, first-order optimization, and Markov chains. These techniques carry over to more complex problems.