import numpy as np
import matplotlib.pyplot as plt

dqn = np.load("dqn_losses_100.npy")*10
dqn = dqn.reshape(10,100)
dqn = dqn.mean(axis=0)
plt.plot(dqn, label="DQN", linewidth=3)

ddqn = np.load("ddqn_losses_100.npy")*10
ddqn = ddqn.reshape(10,100)
ddqn = ddqn.mean(axis=0)
plt.plot(ddqn, label='DDQN', linewidth=3)

dueling_ddqn = np.load("dueling_ddqn_losses_100.npy")*10
print(dueling_ddqn)
dueling_ddqn = dueling_ddqn.reshape(10,100)
dueling_ddqn = dueling_ddqn.mean(axis=0)
plt.plot(dueling_ddqn, label='Dueling-DDQN', linewidth=3)

plt.xlabel("Episodes")
plt.ylabel("Mean reward")
plt.legend()
plt.show()
