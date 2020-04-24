import numpy as np
import random

from gym.spaces import Box

class IoT_Simulation(object):

    def __init__(self, probabilities=[], devices=[], bandwidth_threshold=100000, device_threshold=10, hvft_apps=6, hvft_sizes=[]):
        self.hvft_apps = hvft_apps

        self.action_space = Box(low=0, high=1440, dtype=np.int32, shape=(1, self.hvft_apps))
        self.observation_space = Box(low=0, high=np.inf, dtype=np.int32, shape=(1, 1440))

        self.state = np.zeros((1, 1440), dtype=np.int32)

        self.episodes = 0

        if hvft_sizes:
            self.hvft_sizes = hvft_sizes
        else:
            self.hvft_sizes = [bandwidth_threshold * 0.4] * hvft_apps

        if probabilities:
            self.probs = probabilities
        else:
            self.probs = [0.05] * 2 + [0.03] + [0.02] * 2 + [0.03]  + [0.05] + [0.08] * 4 + [0.1] * 2 + [0.15] * 5 + [0.1] * 2 + [0.08] * 4
        
        if devices:
            self.devices = devices
        else:
            self.devices = [10000, 5000, 3000, 6000, 1000, 500, 2000, 20000, 600, 1500, 3000, 1200, 12000, 20000, 10000, 3000, 2000, 9000, 100, 200]
        
        self.bandwidth_threshold = bandwidth_threshold
        self.device_threshold = device_threshold
        self.reset()
    
    def seed(self, rnd_seed):
        random.seed(rnd_seed)

    def reset(self):
        self.state, _ = self.generate_network_timeseries()
        return self.state

    def generate_network_timeseries(self):
        bandwidth_timeseries = [0] * 1440
        device_timeseries = [0] * 1440

        for i in range(24):
            prob = self.probs[i]
            for j in range(20):
                for device in self.devices:
                    if random.random() <= prob:
                        for k in range(3):
                            bandwidth_timeseries[i * 60 + j*3 + k] += device
                            device_timeseries[i * 60 + j*3 + k] += 1
        
        return bandwidth_timeseries, device_timeseries

    
    def update_state(self, new_series):
        # for i in range(len(self.state)-1, 1):
        #     self.state[i] = self.state[i-1]
        # self.state[0] = new_series
        self.state = np.array(new_series)
    
    def step(self, action):
        
        bandwidth_timeseries, device_timeseries = self.generate_network_timeseries()
        for i in range(len(action[0])):
            timestamp = int(action[0][i])
            for j in range(60):
                bandwidth_timeseries[timestamp + j] += self.hvft_sizes[i]
                device_timeseries[timestamp + j] += 1
        
        band_array = np.array(bandwidth_timeseries)
        device_array = np.array(device_timeseries)

        band_reward = np.average(self.bandwidth_threshold - band_array)
        device_reward = np.average(self.device_threshold - device_array)

        reward = band_reward * 0.75 + device_reward * 0.25

        self.update_state(band_array)

        self.episodes += 1

        if self.episodes > 100:
            done = True
        else:
            done = False

        return self.state, reward, done
        
    
