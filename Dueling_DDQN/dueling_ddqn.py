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



class DuelingDQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).float().unsqueeze(0)
            qvals = self.forward(state)
            action = np.argmax(qvals.cpu().detach().numpy())
        return action



class DuelingDDQNAgent:

    def __init__(self, env, learning_rate=1e-4, gamma=0.99, buffer_size=10000, hidden_dim=128, tau=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingDQN(env.observation_space.shape[0], hidden_dim, env.action_space.n).to(self.device)
        self.target_model = DuelingDQN(env.observation_space.shape[0], hidden_dim, env.action_space.n).to(self.device)

        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

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
        next_Q = self.target_model.forward(next_states)
        next_Q_model = self.model.forward(next_states)
        next_actions = torch.argmax(next_Q_model,1)
        next_Q_expected = next_Q.index_select(1, next_actions)
        expected_Q = rewards + (1-dones)*self.gamma*next_Q_expected

        loss = self.MSE_loss(current_Q, expected_Q.detach())
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1-self.tau)*target_param)

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
    eps = 1

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        eps -= 1/max_episodes
        epsilon = max(eps, 0.05)

        for step in range(max_steps):
            action = agent.get_action(state, eps=epsilon)
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

        if agent.eval_dqn()>400:
            print("You reached a reward over 500. Training Stopped.")
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
    MAX_EPISODES = 400 
    MAX_STEPS = 200
    BATCH_SIZE = 64

    env = gym.make("MountainCar-v0")
    #env._max_episode_steps = 501 

    dueling_ddqn = DuelingDDQNAgent(env, gamma=0.999, learning_rate=0.0005)
    mini_batch_train(env, dueling_ddqn, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

    test(dueling_ddqn.model, "MountainCar-v0")



