
import tensorflow as tf
import numpy as np
import gym
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise

from simulation import IoT_Simulation

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.00005

# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE =  0.001

# Soft target update param
TAU = 0.001

RANDOM_SEED = 1234
EXPLORE = 60
DEVICE = '/cpu:0'

class DDPG_Trainer:

    def train(self, env, epochs=1, MINIBATCH_SIZE=30, GAMMA = 0.99, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=10000):
        
        with tf.Session() as sess:
            
            # configuring the random processes
            np.random.seed(RANDOM_SEED)
            tf.set_random_seed(RANDOM_SEED)
            env.seed(RANDOM_SEED)
            
            # info of the environment to pass to the agent
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.shape[1]
            action_bound = np.float64(720)
            
            # Creating agent
            ruido = OUNoise(action_dim, mu = 0.4) # this is the Ornstein-Uhlenbeck Noise
            actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
            critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), DEVICE)

            sess.run(tf.global_variables_initializer())
            
            # Initialize target network weights
            actor.update_target_network()
            critic.update_target_network()
            
            # Initialize replay memory
            replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

            rewards = []

            for i in range(epochs):

                state = env.reset()
                state = np.hstack(state)
                ep_reward = 0
                done = False
                step = 0
                epsilon -= (epsilon/EXPLORE)
                epsilon = np.maximum(min_epsilon,epsilon)

                while (not done):
                        
                    action_original = actor.predict(np.reshape(state, (1, state_dim)))
                    action = action_original + max(epsilon, 0)*ruido.noise()

                    print(f"Step: {step}")

                    print(f"Action: {action}")
                    
                    next_state, reward, done = env.step(action)

                    print(f"Reward: {reward}")

                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                        done, np.reshape(next_state, (actor.s_dim,)))

                    if replay_buffer.size() > MINIBATCH_SIZE:

                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        # Calculate targets
                        
                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])

                        critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                        
                        a_outs = actor.predict(s_batch)
                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])

                        # Update target networks
                        actor.update_target_network()
                        critic.update_target_network()

                
                    state = next_state

                    ep_reward = ep_reward + reward
                    rewards.append(reward)
                    step +=1
                
                if done:
                    ruido.reset() 
                
                if np.average(rewards[-10:]) > -200:
                    print(f"Number of steps: {step + i*100}")
                    return rewards
        return rewards
