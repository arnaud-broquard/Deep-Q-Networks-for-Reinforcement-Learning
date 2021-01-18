import sys
sys.path.append("../")

import numpy as np
import Deep_Q_Learning.dqn as DQN
import Double_DQN.double_dqn as DDQN
import Dueling_DDQN.dueling_ddqn as DuelingDDQN
import gym

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
                episode_reward = agent.eval_dqn(n_sim=1)
                episode_rewards.append(episode_reward)
                if episode%10==0:
                    print("Episode " + str(episode) + ": " + str(agent.eval_dqn()))
                break

            state = next_state

    return episode_rewards


if __name__ == "__main__":
    MAX_EPISODES = 100
    MAX_STEPS = 700
    BATCH_SIZE = 64

    env = gym.make("CartPole-v0")
    env._max_episode_steps = 800
    dqn_losses = []
    ddqn_losses = []
    dueling_ddqn_losses = []

    NB_TEST = 10 
    for i in range(NB_TEST):
        dqn = DQN.DQNAgent(env)
        ddqn = DDQN.DoubleDQNAgent(env)
        dueling_ddqn = DuelingDDQN.DuelingDDQNAgent(env)
        dqn_loss = mini_batch_train(env, dqn, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
        ddqn_loss = mini_batch_train(env, ddqn, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)
        dueling_ddqn_loss = mini_batch_train(env, dueling_ddqn, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

        dqn_losses += dqn_loss
        ddqn_losses += ddqn_loss
        dueling_ddqn_losses += dueling_ddqn_loss

    dqn_losses = np.array(dqn_losses)/NB_TEST
    ddqn_losses = np.array(ddqn_losses)/NB_TEST
    dueling_ddqn_losses = np.array(dueling_ddqn_losses)/NB_TEST

    np.save('dqn_losses_100', dqn_losses)
    np.save('ddqn_losses_100', ddqn_losses)
    np.save('dueling_ddqn_losses_100', dueling_ddqn_losses)

    print(dqn_losses)
    print(ddqn_losses)
    print(dueling_ddqn_losses)



