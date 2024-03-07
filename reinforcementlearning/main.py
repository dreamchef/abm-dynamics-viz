import gymnasium
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

# Create the Lunar Lander environment

env = gymnasium.make('LunarLander-v2')

# Define the Q-network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(env.observation_space.shape[0],)),
    Dense(64, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

# Define the agent
memory = SequentialMemory(limit=100000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=0.1, value_test=0.05, nb_steps=10000)
dqn = DQNAgent(model=model, memory=memory, policy=policy,
               nb_actions=env.action_space.n, nb_steps_warmup=1000,
               target_model_update=1e-2, enable_double_dqn=True)
dqn.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Evaluate the agent
dqn.test(env, nb_episodes=10, visualize=True)