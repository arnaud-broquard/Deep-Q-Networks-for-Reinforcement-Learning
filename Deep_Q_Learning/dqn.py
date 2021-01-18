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

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.memory, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).float().unsqueeze(0)
            qvals = self.forward(state)
            action = np.argmax(qvals.cpu().detach().numpy())
        return action



class DQNAgent:

    def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=10000, hidden_dim=128):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(env.observation_space.shape[0], hidden_dim, env.action_space.n).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.MSE_loss = nn.MSELoss()


    def get_action(self, state, eps=0.2):
        if np.random.random() < eps:
            return self.env.action_space.sample()

        with torch.no_grad():
            state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
            qvals = self.model.forward(state)
            action = np.argmax(qvals.cpu().detach().numpy())

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_state, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_state).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        current_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + (1-dones)*self.gamma*max_next_Q

        loss = self.MSE_loss(current_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def eval_dqn(self, n_sim=5):
        env_copy = deepcopy(self.env)
        episode_rewards = np.zeros(n_sim)
        for ii in range(n_sim):
          state = env_copy.reset()
          done = False
          while not done:
            action = self.get_action(state, 0.0)
            next_state, reward, done, _ = env_copy.step(action)
            state = next_state
            episode_rewards[ii] += reward

        return episode_rewards.mean()



def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    eps=1

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        eps = max((1-1/(max_episodes))*eps,0.05) 

        for step in range(max_steps):
            action = agent.get_action(state, eps=eps)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                if episode%10==0:
                    print("Episode " + str(episode) + ": " + str(agent.eval_dqn()))
                break

            state = next_state

        if agent.eval_dqn()>500:
            break

    return episode_rewards


def test(trained_net, env_name):
    """Show the results of the network in dedicated window

    Args:
        trained_net: Trained network we want to vizualize 
        env_name: name of the env the trained network is made for 
    """
    env = gym.make(env_name)
    obs = env.reset()
    while True:
        env.reset()
        for _ in range(400):
            action = trained_net.get_action(obs)
            obs,_,_,_ = env.step(action)
            env.render()

    
if __name__=="__main__":
    MAX_EPISODES = 300
    MAX_STEPS = 500
    BATCH_SIZE = 256

    env = gym.make("CartPole-v0")
    env._max_episode_steps = 800

    dqn = DQNAgent(env)
    mini_batch_train(env, dqn, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

    test(dqn.model, "CartPole-v0")
