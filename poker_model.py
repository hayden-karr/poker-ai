# poker_model.py
import tensorflow as tf 
from tensorflow.keras import layers, models

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
    value_head = layer.Dense(64, activation='relu')(x)
    value_output = layers.Dense(inputs=inputs, outputs=[policy_output, value_output])

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