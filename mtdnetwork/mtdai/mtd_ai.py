import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, ReLU, BatchNormalization, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import random
from collections import deque

# Define the neural network architecture
def create_network(state_size, action_size, time_series_size):
    # Static feature extraction module
    static_input = Input(shape=(state_size,))
    x = Dense(128)(static_input)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Time-series analysis module
    time_series_input = Input(shape=(time_series_size, 1))
    y = LSTM(64, return_sequences=True)(time_series_input)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = LSTM(32)(y)
    y = ReLU()(y)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)

    # Feature fusion module
    z = Concatenate()([x, y])
    z = Dense(64)(z)
    z = ReLU()(z)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)

    # Q-Network output layer
    output = Dense(action_size)(z)

    model = Model(inputs=[static_input, time_series_input], outputs=output)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return model

# Define a function to update the target network
def update_target_model(target_network, main_network):
    target_network.set_weights(main_network.get_weights())

# Function to act based on model's output
def choose_action(state, time_series, main_network, action_size, epsilon):
    state = state.reshape((1,-1))
    time_series = time_series.reshape((1,-1))

    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = main_network.predict([state, time_series])
    return np.argmax(act_values[0])

# Learning function
def soft_update_target_model(target_network, main_network, tau=0.1):
    main_weights = np.array(main_network.get_weights())
    target_weights = np.array(target_network.get_weights())
    target_network.set_weights(tau * main_weights + (1 - tau) * target_weights)

# Double Q-learning
def replay(memory, main_network, target_network, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, train_start):
    if len(memory) < train_start:
        return
    minibatch = random.sample(memory, batch_size)
    for state, time_series, action, reward, next_state, next_time_series, done in minibatch:
        state = state.reshape((1,-1))
        time_series = time_series.reshape((1,-1))
        next_state = next_state.reshape((1,-1))
        next_time_series = next_time_series.reshape((1,-1))

        target = main_network.predict([state, time_series])
        if done:
            target[0][action] = reward
        else:
            t_next_action = np.argmax(main_network.predict([next_state, next_time_series])[0])
            t_next_q = target_network.predict([next_state, next_time_series])[0][t_next_action]
            target[0][action] = reward + gamma * t_next_q

        main_network.fit([state, time_series], target, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def calculate_reward(current_state, current_time_series, next_state, next_time_series):
    reward = 0

    # Parameters to control the scale of reward and penalty
    hcr_weight = 100
    vulnerability_weight = 50
    mttc_weight = 20
    exposure_penalty = 100
    mtd_time_penalty = 50

    # Reward for reducing Host Compromise Ratio
    reward += (current_state[0] - next_state[0]) * hcr_weight

    # Reward for reducing the number of vulnerabilities
    reward += (current_state[1] - next_state[1]) * vulnerability_weight

    # Reward for increasing Mean Time to Compromise
    reward += (next_time_series[1] - next_time_series[1]) * mttc_weight

    # Penalty for increased Attack Path Exposure Score
    reward -= (next_state[2] - current_state[2]) * exposure_penalty

    # Penalty for high Time Since Last MTD
    if next_time_series[2] > current_time_series[2]:
        reward -= (next_time_series[2] - current_time_series[2]) * mtd_time_penalty

    return reward