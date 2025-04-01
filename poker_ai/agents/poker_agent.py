# poker_rl.py
import numpy as np
import tensorflow as tf 
from collections import deque
import random
from enum import Enum
from poker_ai.environment.texas_holdem import Action

from poker_ai.models.policy_model import create_bet_size_model, create_poker_model

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

    def get_bet_size(self, state, max_bet_size):
        """
        Determine bet size when raising based on the current state
        """
        # Default implementation uses a percentage of the max bet size
        # based on the bet sizing model's prediction
        state_tensor = np.array([state])
        bet_percentage = self.bet_model.predict(state_tensor, verbose=0)[0][0]
        
        # Ensure a minimum bet of 1 or 10% of max bet size
        min_bet = max(1, 0.1 * max_bet_size)
        
        # Scale the percentage to an actual bet size
        bet_size = min_bet + bet_percentage * (max_bet_size - min_bet)
        
        return int(bet_size)  # Return an integer bet amount
    
class PokerAgent:
    """Reinforcement learning agent for poker"""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.network = PokerNeuralNetwork(config.model_config)
        self.memory = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_start
        self.steps = 0
        
    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """
        Select an action based on current state
        
        Args:
            state: Current state representation
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random valid action
            return np.random.choice(valid_actions)
        
        # Get action probabilities from policy network
        action_probs, _ = self.network.predict_action(state)
        
        # Filter valid actions
        valid_probs = np.zeros_like(action_probs)
        for action in valid_actions:
            valid_probs[action] = action_probs[action]
            
        # If all probabilities are zero, choose randomly
        if np.sum(valid_probs) == 0:
            return np.random.choice(valid_actions)
            
        # Normalize probabilities
        valid_probs = valid_probs / np.sum(valid_probs)
        
        # Choose action with highest probability
        return np.argmax(valid_probs)
    
    def get_bet_size(self, state: np.ndarray, max_bet: int) -> int:
        """
        Determine bet size when raising
        
        Args:
            state: Current state representation
            max_bet: Maximum possible bet
            
        Returns:
            Bet amount
        """
        bet_pct = self.network.predict_bet_size(state)
        
        # Minimum bet is either 1 or a percentage of max bet
        min_bet = max(1, int(0.1 * max_bet))
        
        # Scale bet percentage to actual amount
        bet_amount = min_bet + int(bet_pct * (max_bet - min_bet))
        
        return bet_amount
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
        
    def update_epsilon(self) -> None:
        """Update exploration rate"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
    def train(self) -> Dict[str, float]:
        """
        Train the agent using experience replay
        
        Returns:
            Dictionary with loss information
        """
        # Skip if not enough samples
        if len(self.memory) < self.config.batch_size:
            return {'policy_loss': 0, 'value_loss': 0}
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        # Get current and next state values
        _, values = self.network.policy_model.predict(states, verbose=0)
        _, next_values = self.network.policy_model.predict(next_states, verbose=0)
        
        # Calculate target values (Bellman equation)
        target_values = rewards + self.config.gamma * next_values.flatten() * (1 - dones)
        
        # One-hot encode actions
        action_one_hot = tf.one_hot(actions, self.config.action_dim)
        
        # Custom training step
        with tf.GradientTape() as tape:
            # Forward pass
            policy_pred, value_pred = self.network.policy_model(states, training=True)
            
            # Calculate losses
            policy_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    action_one_hot, policy_pred
                )
            )
            
            value_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    target_values, value_pred[:, 0]
                )
            )
            
            # Combined loss
            total_loss = policy_loss + value_loss
        
        # Calculate gradients and update weights
        grads = tape.gradient(total_loss, self.network.policy_model.trainable_variables)
        self.network.policy_model.optimizer.apply_gradients(
            zip(grads, self.network.policy_model.trainable_variables)
        )
        
        # Update exploration rate
        self.update_epsilon()
        self.steps += 1
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'epsilon': self.epsilon
        }
        
    def train_bet_model(self, states: List[np.ndarray], bet_sizes: List[int], 
                        max_bets: List[int]) -> Dict[str, float]:
        """
        Train the bet sizing model
        
        Args:
            states: List of state representations
            bet_sizes: List of bet amounts
            max_bets: List of maximum possible bets
            
        Returns:
            Dictionary with loss information
        """
        # Normalize bet sizes
        normalized_bets = np.array([
            bet / max_bet for bet, max_bet in zip(bet_sizes, max_bets)
        ]).reshape(-1, 1)
        
        # Train model
        history = self.network.bet_model.fit(
            np.array(states),
            normalized_bets,
            epochs=5,
            batch_size=min(32, len(states)),
            verbose=0
        )
        
        return {'bet_loss': history.history['loss'][-1]}
    
    def save(self, directory: str) -> None:
        """
        Save agent to directory
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save neural network
        self.network.save(os.path.join(directory, 'network'))
        
        # Save agent configuration
        agent_config = {
            'state_dim': self.config.state_dim,
            'action_dim': self.config.action_dim,
            'gamma': self.config.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.config.epsilon_end,
            'epsilon_decay': self.config.epsilon_decay,
            'steps': self.steps
        }
        
        with open(os.path.join(directory, 'agent_config.json'), 'w') as f:
            json.dump(agent_config, f)
    
    @classmethod
    def load(cls, directory: str) -> 'PokerAgent':
        """
        Load agent from directory
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded PokerAgent instance
        """
        # Load neural network
        network = PokerNeuralNetwork.load(os.path.join(directory, 'network'))
        
        # Load agent configuration
        with open(os.path.join(directory, 'agent_config.json'), 'r') as f:
            agent_config = json.load(f)
        
        # Create config objects
        model_config = ModelConfig(
            input_dim=agent_config['state_dim'],
            num_actions=agent_config['action_dim']
        )
        
        config = AgentConfig(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            gamma=agent_config['gamma'],
            epsilon_start=agent_config['epsilon'],
            epsilon_end=agent_config['epsilon_end'],
            epsilon_decay=agent_config['epsilon_decay'],
            model_config=model_config
        )
        
        # Create agent
        agent = cls(config)
        agent.network = network
        agent.epsilon = agent_config['epsilon']
        agent.steps = agent_config['steps']
        
        return agent