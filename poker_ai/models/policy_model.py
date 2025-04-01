# poker_model.py
import tensorflow as tf 
from tensorflow.keras import layers, models
import os

def create_poker_model(input_dim, num_actions):
    """
    Create a nueral netweork for the poker-playing ai.
    It combines two networks:
    1. Policy network - decides which action to take
    2. Value network - estimates how good the current state is 
    """
    # Common layers for feature extraction
    inputs = layers. Input(shape=(input_dim,))

    # Initial dense layers with normalization and activation
    x = layers.Dense(256)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Policy head ( action probabilities)
    policy_head = layers.Dense(64,activation='relu')(x)
    policy_output = layers.Dense(num_actions, activation='softmax', name='policy')(policy_head)

    # Value head (state value estimations)
    value_head = layers.Dense(64, activation='relu')(x)  
    value_output = layers.Dense(1, name='value')(value_head)

    # Create model
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])

    # Compile model with appropriate losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squarred_error'
        }
    )

    return model

def create_bet_size_model(input_dim):
    """
    Create a seperate model to decide bet size when raising
    """
    inputs =layers.Input(shape=(input_dim,))

    x= layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)

    # Output is normalized bet size (0-1 range)
    output = layers.Dense(1, activition='sigmoid', name='bet_size')(x)

    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )

    return model

class PokerNeuralNetwork:
    """Neural network for poker decision making"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the neural network
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.policy_model = self._build_policy_model()
        self.bet_model = self._build_bet_model()
        
    def _build_policy_model(self) -> tf.keras.Model:
        """
        Build the policy and value network
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.config.input_dim,))
        
        # Hidden layers with batch normalization and dropout
        x = inputs
        for units in self.config.hidden_layers:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Policy head
        policy_head = layers.Dense(64, activation='relu')(x)
        policy_output = layers.Dense(
            self.config.num_actions, 
            activation='softmax', 
            name='policy'
        )(policy_head)
        
        # Value head
        value_head = layers.Dense(64, activation='relu')(x)
        value_output = layers.Dense(1, name='value')(value_head)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
        
        # Compile with appropriate loss functions
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            }
        )
        
        return model
    
    def _build_bet_model(self) -> tf.keras.Model:
        """
        Build the bet sizing model
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.config.input_dim,))
        
        # Hidden layers
        x = inputs
        for units in self.config.hidden_layers[:2]:  # Use fewer layers for bet model
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output is normalized bet size (0-1)
        output = layers.Dense(1, activation='sigmoid', name='bet_size')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=output)
        
        # Compile with MSE loss
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mean_squared_error'
        )
        
        return model
    
    def predict_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict action probabilities and state value
        
        Args:
            state: State representation
            
        Returns:
            Tuple of (action probabilities, state value)
        """
        state_tensor = np.expand_dims(state, axis=0)
        action_probs, value = self.policy_model.predict(state_tensor, verbose=0)
        return action_probs[0], value[0][0]
    
    def predict_bet_size(self, state: np.ndarray) -> float:
        """
        Predict bet size as percentage of max bet
        
        Args:
            state: State representation
            
        Returns:
            Bet size percentage (0-1)
        """
        state_tensor = np.expand_dims(state, axis=0)
        bet_pct = self.bet_model.predict(state_tensor, verbose=0)[0][0]
        return bet_pct
    
    def save(self, directory: str) -> None:
        """
        Save models to directory
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save policy model
        self.policy_model.save(os.path.join(directory, 'policy_model.h5'))
        
        # Save bet model
        self.bet_model.save(os.path.join(directory, 'bet_model.h5'))
        
        # Save configuration
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump({
                'input_dim': self.config.input_dim,
                'num_actions': self.config.num_actions,
                'hidden_layers': self.config.hidden_layers,
                'dropout_rate': self.config.dropout_rate,
                'learning_rate': self.config.learning_rate
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'PokerNeuralNetwork':
        """
        Load models from directory
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded PokerNeuralNetwork instance
        """
        # Load configuration
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(**config_dict)
        
        # Create instance
        instance = cls(config)
        
        # Load models
        instance.policy_model = models.load_model(
            os.path.join(directory, 'policy_model.h5')
        )
        
        instance.bet_model = models.load_model(
            os.path.join(directory, 'bet_model.h5')
        )
        
        return instance