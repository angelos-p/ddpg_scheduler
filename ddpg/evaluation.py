from simulation import IoT_Simulation
from ddpg import DDPG_Trainer
from scipy.interpolate import BSpline, make_interp_spline

import time
import numpy as np
import matplotlib.pyplot as plt


def smooth_plot(x, y):
    x = np.array(x)
    y = np.array(y)

    x_new = np.linspace(x.min(), x.max(), 300)
    a_BSpline = make_interp_spline(x, y)
    y_new = a_BSpline(x_new)

    return x_new, y_new

def create_results_graph(env, hvft_size_fraction):
    start = time.time()
    trainer = DDPG_Trainer()

    rewards = trainer.train(env)
    label = f"{hvft_size_fraction} hvft fraction size"
    plt.plot(range(len(rewards)), rewards, label=label)

    end = time.time()

    return end - start


iot = IoT_Simulation()
band_timeseries, device_timeseries = iot.generate_network_timeseries()

# plt.plot(range(len(band_timeseries)), band_timeseries)
# plt.plot(range(len(device_timeseries)), device_timeseries)
# plt.show()

# times = []
# for i in [0.2, 0.25, 0.3, 0.35]:
#     env = IoT_Simulation(hvft_apps=7, hvft_size_fraction=i)
#     times.append(create_results_graph(env))

times = []
devices_traffic = list(np.linspace(5000,10000,150))
for i in [0.06, 0.08, 0.1, 0.13]:
    env = IoT_Simulation(devices_traffic=devices_traffic, bandwidth_threshold=1000000, device_threshold=120, hvft_apps=30, hvft_size_fraction=i)
    times.append(create_results_graph(env, i))

plt.xlabel("Step number")
plt.ylabel("Reward")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

# plt.bar(range(len(times)), times)
# plt.xticks(range(len(times)), ("0.8 fraction", "0.1 fraction", "0.13 fraction", "0.15 fraction"))
# plt.ylabel("Training Time (s)")

plt.show()