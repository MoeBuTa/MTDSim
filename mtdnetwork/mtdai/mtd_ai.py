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





def normalize_array(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr


def calculate_reward(current_state, current_time_series, next_state, next_time_series, static_features, time_features, memory):
    reward = 0

    # Check if memory has data for normalization
    if len(memory) > 0:
        # Extract min and max values for normalization from memory
        memory_states = [item[0] for item in memory]
        memory_time_series = [item[5] for item in memory]
        
        # Flatten lists of arrays to compute overall min and max
        all_states = np.concatenate(memory_states, axis=0)
        all_time_series = np.concatenate(memory_time_series, axis=0)

        # Determine min and max for normalization
        min_state, max_state = np.min(all_states), np.max(all_states)
        min_time_series, max_time_series = np.min(all_time_series), np.max(all_time_series)

        # Normalize current and next states
        norm_current_state = normalize_array(current_state, min_state, max_state)
        norm_next_state = normalize_array(next_state, min_state, max_state)

        # Normalize current and next time series
        norm_current_time_series = normalize_array(current_time_series, min_time_series, max_time_series)
        norm_next_time_series = normalize_array(next_time_series, min_time_series, max_time_series)
    else:
        # Use raw values if memory is empty
        norm_current_state = current_state
        norm_next_state = next_state
        norm_current_time_series = current_time_series
        norm_next_time_series = next_time_series
    # print(norm_current_state,norm_current_time_series)
    # Dynamic weights based on context
    context_multiplier = 1  # Adjust this dynamically based on system context
    dynamic_weights = {
        "host_compromise_ratio": -75 * context_multiplier,
        "exposed_endpoints": -75 * context_multiplier,
        "attack_path_exposure": -75 * context_multiplier,
        "overall_asr_avg": 75 * context_multiplier,
        "roa": 75 * context_multiplier,
        "shortest_path_variability": 75 * context_multiplier,
        "risk": -75 * context_multiplier,
        "attack_type": 0
    }

    # Include time series features in the dynamic weights
    time_series_weights = {
        "mtd_freq": 20 * context_multiplier,
        "overall_mttc_avg": 75 * context_multiplier,
        "time_since_last_mtd": -75 * context_multiplier
    }

    # Calculate reward using normalized or raw values
    for index, feature in enumerate(static_features):
        delta = (norm_next_state[index] - norm_current_state[index])
        reward += delta * dynamic_weights.get(feature, 0)
     
    for index, time_series_feature in enumerate(time_features):
        delta = (norm_next_time_series[index] - norm_current_time_series[index])
        reward += delta * time_series_weights.get(time_series_feature, 0)
  
    return reward


