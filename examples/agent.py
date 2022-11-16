"""Training the agent"""

import random
import numpy as np
import gymnasium as gym

print(gym.__version__)

env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(0, 100_000):
    state, info = env.reset()

    epochs, penalties, reward, = (0, 0, 0)
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        print(f"Episode: {i} in {epochs}")

print("Training finished.\n")
