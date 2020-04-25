import numpy as np
import random

import matplotlib.pyplot as plt

from gym.spaces import Box

class IoT_Simulation(object):

    def __init__(self, probabilities=[], devices_traffic=[], bandwidth_threshold=120000, device_threshold=15, hvft_apps=6, hvft_sizes=[], hvft_size_fraction=0.25, epoch_size=100):
        
        self.hvft_apps = hvft_apps
        self.epoch_size = 100
        self.action_space = Box(low=0, high=1440, dtype=np.int32, shape=(1, self.hvft_apps))
        self.observation_space = Box(low=0, high=np.inf, dtype=np.int32, shape=(1, 1440))

        self.state = np.zeros((1, 1440), dtype=np.int32)

        self.episodes = 0

        if hvft_sizes:
            self.hvft_sizes = hvft_sizes
        else:
            self.hvft_sizes = [bandwidth_threshold * hvft_size_fraction] * hvft_apps

        # Probabilities (D_on, D_off)
        if probabilities:
            self.probs = probabilities
        else:
            self.probs = [(0.1, 0.9)] * 2 + [(0.1, 0.9)] + [(0.07, 0.9)] * 2 + [(0.05, 0.9)]  + [(0.07, 0.9)] + [(0.1, 0.8)] * 4 + [(0.2, 0.8)] * 2 + [(0.3, 0.8)] * 5 + [(0.2, 0.8)] * 2 + [(0.15, 0.8)] * 4
        
        if devices_traffic:
            self.devices_traffic = devices_traffic
        else:
            self.devices_traffic = [10000, 5000, 3000, 6000, 1000, 5000, 2000, 20000, 6000, 1500, 3000, 1200, 12000, 20000, 10000, 3000, 2000, 9000, 1000, 2000]
        
        self.device_on = [False] * len(self.devices_traffic)
        
        self.bandwidth_threshold = bandwidth_threshold
        self.device_threshold = device_threshold
        self.reset()
    
    def seed(self, rnd_seed):
        random.seed(rnd_seed)

    def reset(self):
        self.state, _ = self.generate_network_timeseries()
        self.episodes = 0
        return self.state

    def generate_network_timeseries(self):
        bandwidth_timeseries = [0] * 1440
        device_timeseries = [0] * 1440

        for i in range(24):
            prob = self.probs[i]
            for j in range(20):
                for d in range(len(self.devices_traffic)):
                    if self.device_on[d]:
                        if random.random() > prob[1]:
                            for k in range(3):
                                bandwidth_timeseries[i * 60 + j*3 + k] += self.devices_traffic[d]
                                device_timeseries[i * 60 + j*3 + k] += 1
                        else:
                            self.device_on[d] = False
                    else:
                        if random.random() < prob[0]:
                            self.device_on[d] = True
                            for k in range(3):
                                bandwidth_timeseries[i * 60 + j*3 + k] += self.devices_traffic[d]
                                device_timeseries[i * 60 + j*3 + k] += 1 
        
        return bandwidth_timeseries, device_timeseries

    
    def update_state(self, new_series):
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

        band_reward = - np.average(np.maximum(band_array - self.bandwidth_threshold, 0))
        device_reward = - np.average(np.maximum(device_array - self.device_threshold, 0))


        reward = band_reward * 0.8 + device_reward * 0.2 * (self.bandwidth_threshold/self.device_threshold)

        self.update_state(band_array)

        self.episodes += 1

        if self.episodes > self.epoch_size:
            done = True
        else:
            done = False

        return self.state, reward, done