from simulation import IoT_Simulation
from ddpg import DDPG_Trainer

import numpy as np
import matplotlib.pyplot as plt

devices_traffic = list(np.linspace(5000,10000,150))
env = IoT_Simulation(devices_traffic=devices_traffic, bandwidth_threshold=1000000, device_threshold=120, hvft_apps=30, hvft_size_fraction=0.08)

trainer = DDPG_Trainer()
rewards = trainer.train(env)

plt.plot(range(len(rewards)), rewards)
plt.show()