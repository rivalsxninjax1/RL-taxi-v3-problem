import gym
import numpy as np
import random

# Create the Taxi environment
env = gym.make("Taxi-v3")

# Initialize Q-table with zeros.
# The number of states and actions are based on the environment's observation and action space.
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1          # Learning rate
gamma = 0.6          # Discount factor for future rewards
epsilon = 0.1        # Exploration rate
num_episodes = 1000  # Number of training episodes

# Training the agent using Q-learning
for episode in range(num_episodes):
    state = env.reset()  # Reset environment to initial state for each episode
    done = False

    while not done:
        # Choose an action (explore or exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Exploration: random action
        else:
            action = np.argmax(q_table[state])   # Exploitation: best known action

        # Perform the chosen action and receive new state and reward
        next_state, reward, done, info = env.step(action)

        # Q-learning update rule:
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s,a)]
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        # Move to the next state
        state = next_state

# After training, we test the agent on one episode:
state = env.reset()
done = False
total_reward = 0

print("Testing the trained agent:")
while not done:
    # Choose the best action from the Q-table
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render()  # Render the environment (console-based visualization)
    
print("Total reward received:", total_reward)
env.close()
