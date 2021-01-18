import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gym

class ReplayBuffer:
    """
    Replay buffer that memorize previous events
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return rng.choice(self.memory, batch_size).tolist()


    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Network that store the estimate of the Q-function
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def get_q(env, states):
    """
    Compute Q function for a list of states
    """
    with torch.no_grad():
        states_v = torch.FloatTensor([states])
        output = q_net.forward(states_v).data.numpy() 
    return output[0, :, :]  


def choose_action(env, state, epsilon):
    """
    Return action according to an epsilon-greedy exploration policy
    """
    if epsilon>np.random.uniform():
      action = env.action_space.sample()
    else:
      q = get_q([state])[0]
      action = q.argmax()
    return action


def eval_dqn(env, n_sim=5):
    """
    Monte Carlo evaluation of DQN agent.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for ii in range(n_sim):
      state = env_copy.reset()
      done = False
      while not done:
        action = choose_action(env, state, 0.0)
        next_state, reward, done, _ = env_copy.step(action)
        state = next_state
        episode_rewards[ii] += reward

    return episode_rewards


def update(net, state, action, reward, next_state, done):
    """
    """
    
    # add data to replay buffer
    if done:
        next_state = None
    replay_buffer.push(state, action, reward, next_state)
    
    if len(replay_buffer) < BATCH_SIZE:
        return np.inf
    
    transitions = replay_buffer.sample(BATCH_SIZE)  

    values  = torch.zeros(BATCH_SIZE)   
    targets = torch.zeros(BATCH_SIZE)  
    for ii, transition in enumerate(transitions):
        state, action, reward, next_state = transition

        state_torch = torch.FloatTensor([state])
        reward_torch = torch.FloatTensor([reward])[0]

        Q_si_ai = q_net(state_torch)[0, action]

        max_Q_next_state = 0.0
        if next_state is not None:
          next_state_torch = torch.FloatTensor([next_state])
          max_Q_next_state = target_net(next_state_torch).detach()[0].max()

        yi = reward_torch + GAMMA*max_Q_next_state
        values[ii] = Q_si_ai
        targets[ii] = yi

    objective = nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-2)

    loss = objective(values, targets)   

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.data.numpy()


def train(env, net, target_net, n_episodes=400):
    state = env.reset()
    epsilon = 1.0 
    ep = 0
    total_time = 0
    while ep < n_episodes:
        action = choose_action(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        loss = update(state, action, reward, next_state, done)

        state = next_state

        if done:
            state = env.reset()
            ep   += 1
            if ( (ep+1)% EVAL_EVERY == 0):
                rewards = eval_dqn()
                print("episode =", ep+1, ", reward = ", np.mean(rewards))
                if np.mean(rewards) >= REWARD_THRESHOLD:
                    break

            if ep % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * \
                            np.exp(-1. * ep / DECREASE_EPSILON )  

        total_time += 1

# Run the training loop
train()

# Evaluate the final policy
rewards = eval_dqn(2)
print("")
print("mean reward after training = ", np.mean(rewards))
