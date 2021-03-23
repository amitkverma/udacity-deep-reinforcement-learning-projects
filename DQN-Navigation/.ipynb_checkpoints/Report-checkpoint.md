# Project Navigation - Report
Author: Amit Kumar Verma

## Introduction
The project demonstrates the ability of value-based methods, specifically, [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and its variants, to learn a suitable policy in a model-free Reinforcement Learning setting using a Unity environment, which consists of a continuous state space of 37 dimensions, with the goal to navigate around and collect yellow bananas (reward: +1) while avoiding blue bananas (reward: -1). There are 4 actions to choose from: move left, move right, move forward and move backward. 
This project contains the implemented of improvements over vanilla DQN. 

## Issues with vanilla DQN
1. **Overfiting over expirences** : In vanilla DQN, while training the network we feed them the immediate expirences which are highly co-releated, Network tends to explore less and become bais toward the immediate expirence. ( Solved by adding **replay buffer** )
2. **Moving target** : In vanilla DQN, we use the same network for both choosing the actions and evaluating of actions, this leads to training instability. ( Solved by adding two networks **local** and **target** used in `Double DQN` implementation )
3. **Bais Problem** : Double DQN comes with problem where it tends to overestimate the reward in noisy enviroment, leading to non-optimal training outcomes ( Solved by spliting the output layer in `value stream` and `action adventage stream` used in `Duel DQN` ).

## Learning Algorithm
Below, i have mentioned implementation summery of these networks, more detailed implementation is given in notebook.

- **Double DQN**: The idea of Double DQN is to disentangle the calculation of the Q-targets into finding the best action and then calculating the Q-value for that action in the given state. The trick then is to use one network to choose the best action and the other to evaluate that action. The intuition here is that if one network chose an action as the best one by mistake, chances are that the other network wouldn't have a large Q-value for the sub-optimal action. More details can be found in the notebook `Double_DQN_Navigation.ipynb`.

- **Prioritized DQN Network**: Prioritized DQN is one of the improvements over Double DQNs. The basic idea is that some experiences may be more important than others for our training, but it might occur less frequently. Because we sample the batch randomly these rich experiences will rarely get selected or no chance to be selected. Hence, we add priority with each experience and sample based on priority. Priority is calculated using TD error:
$$\delta_i = r_t + \gamma \max_{a \in \mathcal{A}} Q_{\theta^-}(s_{t+1},a) - Q_\theta(s_t,a_t)$$
Hence, We can calculate the priority  
$$p_i = | \delta_i | + \epsilon$$
More details can be found in the notebook `Prioritized_DQN_Navigation.ipynb`

- **Dueling Network**: Vanilla DQNs have a single output stream with the number of output nodes equal to the number of actions. But this could lead to unnecessarily estimating the value of all the actions for states for states which are clearly bad and where, choosing any action won't matter that much. So, the idea behind dueling networks is to have two output streams, with a shared feature extractor layer. One stream outputs a single scalar value denoting the value function for that state, `V(s)` while the other stream outputs the advantage function for each action in that state `A(a, s)`. The advantage function accounts for the advantage achieved for choosing action `a` . They are combined together using a special aggregrate layer:
$$ Q (s, a) = V(s) + (A(s, a) - 1/A * mean_a (A (s, a))$$
The mean subtraction is done to avoid the identifiability problem. More details can be found in the notebook `Dueling_DQN_Navigation.ipynb`.

## Hyperparameters

# Hyperparameters

| Hyperparameter        | Double DQN | Prioritized DQN | Duel DQN |
|-----------------------|------------|-----------------|----------|
| Replay buffer size    |  1e5       |                 |          |
| Batch size            |  64        | 64              |  64      |
| $\gamma$              |  0.99      | 0.99            |  0.99    |
| $\tau$                |  1e-3      |                 |          |
| Learning rate         |  5e-3      | 5e-3            |  5e-3    |
| update interval       |  4         |                 |          |
| Number of episodes    |  1000      | 1000            |  1000    |
| timesteps per episode |  2000      |                 |          |
| Epsilon start         |  1.0       | 1.0             |  1.0     |
| Epsilon minimum       |  .995      | .995            |  .995    |
| Epsilon decay         |  0.05      | 0.05            |  0.05    |

Note: some of hyperparameter are not mentioned here since they are network specific.

## Plot of Rewards
Plot showing the score per episode over all the episodes. The environment was solved in **1000** episodes.
| Double DQN | Prioritize DQN | Dueling DQN |
:-------------------------:|:-------------------------:|:-------------------------:
![double-dqn-scores](./results/double_dqn_result.png) |  ![prioritize-scores](./results/prioritize_dqn_result.png) | ![dueling-dqn-scores](./results/duel_dqn_result.png) 

## Ideas for Future Work
- Hyperparameter search for both Double DQNs and Dueling Double DQNs should lead to better performance too.
- Using the [Rainbow DQN](https://arxiv.org/abs/1710.02298) performance of the model can be improved even further.
