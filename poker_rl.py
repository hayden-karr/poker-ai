# poker_rl.py
import numpy as np
import tensorflow as tf 
from collections import deque
import random
from enum import Enum
from texas_holdem import Action

from poker_model import create_bet_size_model, create_poker_model

class PokerAgent:
    def __init__(self, state_dim, action_dim, model=None, bet_model=None):
        self.state_dim = state_dim
        self.action_dim = action_dim 

        # Initialize models
        if model is None:
            self.model = create_poker_model(state_dim, action_dim)
        else:
            self.model = model

        if bet_model is None:
            self.bet_model = create_bet_size_model(state_dim)
        else:
            self.bet_model = bet_model

        # Memory for experience replay 
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.95 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def get_action(self, state, valid_actions):
        """
        Select an action based on the current state
        """
        if np.random.rand() < self. epsilon:
            # Explore: choose random valid action
            return random.choice(valid_actions)

        # Exploit: choose best valid actionaccording to model
        action_probs, _ = self.model.predict(state.reshape(1,-1), verbose = 0)

        # Filter only valid actions
        valid_probs = np.zeros_like(action_probs[0])
        for action in valid_actions:
            valid_probs[action.value] = action_probs[0][action.value]

        # Renormalize probabilities 
        if np.sum(valid_probs) > 0:
            valid_probs = valid_probs / np.sum(valid_probs)
        else:
            # If all zero, choose randomly from valid actions 
            valid_probs = np.zeros_like(action_probs[0])
            for action in valid_actions:
                valid_probs[action.value] = 1.0/ len(valid_actions)
        
        return Action(np.argmax(valid_probs))

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Train the model using experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # SImple random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1].value for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Get value predictions for the current and next states
        _, values = self.model.predict(states, verbose=0)
        _, next_values = self.model.predict(next_states, verbose=0)

        # Calculate target values for training
        target_values =rewards + self.gamma * next_values.flatten() * (1- dones)

        # One-hot encode actions
        action_one_hot = tf.one_hot(actions, self.action_dim)

        # Custom training step
        with tf.GradientTape() as tape:
            # Forwar pass
            policy_pred, value_pred = self.model(states, training=True)

            # Calculate Losses
            policy_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(action_one_hot, policy_pred)
            )
            value_loss = tf.reduce_mean(
                tf.keras.losses.mean_squarred_error(target_values, value_pred[:, 0])
            )

            # Combined loss
            total_loss = policy_loss + value_loss

        # Calculate gradients and update weights
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Decay exploaration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def train_bet_model(self, states, bet_sizes, max_bets):
        """
        Train the bet sizing model
        """
        # Normalize bet sizes
        normalized_bets = np.array([bet / max_bet for bet, max_bet in zip(bet_sizes, max_bets)])

        # Train the model
        self.bet_model.fit(
            np.array(states),
            normalized_bets.reshape(-1,1),
            epoch=5,
            batch_size=32,
            verbose=0
        )
    
    def save_models(self, policy_path='poker_policy_model.h5', bet_path='poker_bet_model.h5'):
        """
        Save the models
        """
        self.model.save(policy_path)
        self.model.sav(bet_path)
    
    def load_model(self, policy_path='poker_policy_model.h5', bet_path='poker_bet_model.h5'):
        """
        Load the models
        """
        self.model = tf.keras.models.load_model(policy_path)
        self.bet_model = tf.keras.models.load_model(bet_path)