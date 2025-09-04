import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQN_agent:
    def __init__(self, env, max_steps=24, memory_size=10000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.env = env
        self.state_size = env.reset().shape[0]
        self.action_size = 3 ** env.n_stores
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.max_steps = max_steps

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def decode_action(self, action_idx):
    
     n = self.env.n_stores + 1  # +1 for central warehouse if needed
     action = []
     for _ in range(n):
        action.append(action_idx % 3)
        action_idx //= 3
     return np.array(action)

    def get_action(self, state):
     if np.random.rand() <= self.epsilon:
        action_idx = np.random.randint(self.action_size)
     else:
        q_values = self.model.predict(np.array([state]), verbose=0)
        action_idx = np.argmax(q_values[0])
     return self.decode_action(action_idx)

    def replay(self):
     if len(self.memory) < self.batch_size:
        return
     minibatch = random.sample(self.memory, self.batch_size)
     states = np.array([m[0] for m in minibatch])
     actions = np.array([m[1] for m in minibatch])
     rewards = np.array([m[2] for m in minibatch])
     next_states = np.array([m[3] for m in minibatch])
     dones = np.array([m[4] for m in minibatch])

     target = self.model.predict(states, verbose=0)
     target_next = self.target_model.predict(next_states, verbose=0)
     for i in range(self.batch_size):
        done_flag = np.any(dones[i]) if isinstance(dones[i], (np.ndarray, list)) else dones[i]
        if done_flag:
            target[i][actions[i]] = rewards[i]
        else:
            target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
     self.model.fit(states, target, epochs=1, verbose=0)

     if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    def train(self, n_episodes=1000):
        for e in range(n_episodes):
            state = self.env.reset()
            for time in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            self.replay()
            if e % 10 == 0:
                print(f"Episode {e}/{n_episodes}, Epsilon: {self.epsilon:.2f}")
    
    def update(self, state, action, reward, next_state, done):
    
    # Convert action vector to index if needed
     if isinstance(action, np.ndarray):
        action_idx = 0
        for i, a in enumerate(action):
            action_idx += a * (3 ** i)
     else:
        action_idx = action
     self.remember(state, action_idx, reward, next_state, done)
     self.replay()
     if np.isscalar(done):
      if done:
        self.update_target_model()
     else:
      if np.any(done):
        self.update_target_model()
    def create_plots(self, rewards, epsilons=None, losses=None):
        """
        Plots rewards, epsilon decay, and optionally loss over episodes.
        """
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(rewards, label='Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        if epsilons is not None:
            plt.subplot(2, 1, 2)
            plt.plot(epsilons, label='Epsilon', color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.legend()
        plt.tight_layout()
        plt.show()