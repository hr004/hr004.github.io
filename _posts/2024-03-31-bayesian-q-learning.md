---
title: Bayesian Q learning
author: Hafizur Rahman
# date: 2020-09-09 11:33:00 +0800
categories: [Bayesian Methods]
tags: [Reinforcement Learning]
math: true
---

Balancing the exploration and exploitation in reinforcement learning. If the agent does not explore enough, it may settle on a policy with low quality. On the other hand, unable to avoiding suboptimal options may result in overexploration and waste of compute resources. It is challenging to find a good trade-off between exploration and exploitation.

## Q learning

Q-learning is a model free reinforcement learning algorithm to learn the quality of the actions. We can think of it as a method of learning a value function, which is a measure of the quality of an action in a given state. The Q-learning algorithm iteratively updates this value function until it converges to the true value function, which gives the optimal action for each state. This is achieved by using a balance of exploration, where the agent tries new actions to gather information, and exploitation, where the agent uses the information it has to choose the best action. The Q-values are updated using the Bellman equation.

- The standard Bellman equations are defined as
  - $V^{*}(s) = \max_{a} Q^{*}(s,a)$
  - $Q^{*}(s,a) = r(s,a) + \gamma \sum_{s'} P(s' | s,a) V^{*}(s')$

- Q-learning algorithm:
  1. Let the current state be $s$.
  2. Select an action $a$ to perform. This can be done using a policy derived from Q, such as $\epsilon$-greedy.
  3. Let the reward received for performing $a$ be $r$ and the resulting state be $s'$.
  4. Update $Q(s,a)$ to reflect the observations $s,a,r,s'$ as follows
     - $Q(s,a) = Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

     where:
     - $\alpha$ is the learning rate
     - $\gamma$ is the discount factor
     - $\text{max} Q(s', a')$ is the maximum Q-value over all possible actions in the new state (s')
  5. Repeat above steps until convergence or maximum number of episodes is reached.

Two most commonly used techniques to balance the exploration and exploitation when choosing action are-
- Semi uniform random exploration
    1. Choose the action with highest return with probability $p$
    2. Choose a random action with probability $1-p$
- Boltzman exploration: It is similar to softmax distribution with temperature parameter to control the exploration. Lowering the value of $T$ over time decreases exploration.

In both the approach no exploration specific knowledge is used means they are undirected.

## Bayesian Q learning

In Q-Learning, point estimates of the Q-values are used to make decisions.
However, point estimates do not capture the uncertainty in the Q-values.
To overcome this limitation, Bayesian Q-Learning uses probability distributions to represent the Q-value of each state which enables us to capture uncertainty in the Q-values which in turn helps the agent to make more informed decision. Similar to undirected exploration, in Bayesian Q-learning, actions are chosen based on the local Q-value information. As we are propagating the distribution over Q-values, local Q-value information contains the information on uncertainty of other states.

### Q-value distribution

We can denote total discounted rewared in state $s$ for action $a$ as $R_{s,a}$ which is a random variable. It is fair to assume that $R_{s,a}$ is normally distributed. This can be justified by the central limit theorem  and Markov chain convergence theorem. If discount factor $\gamma \rightarrow 1$ and MDP is ergodic meaning stationary Markov chain, then applying the optimal policy will make $R_{s,a}$ to be a normal distribution.

