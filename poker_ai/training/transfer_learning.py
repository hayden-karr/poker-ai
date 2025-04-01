import tensorflow as tf
from tensorflow.keras import models, layers

def create_transfer_learning_model(pretrained_model_path, state_dim, action_dim, trainable_layers=2):
    """
    Create a model for transfer learning based on a pre-trained poker model
    """
    # Load pre trained model
    base_model = tf.keras.models.load_model(pretrained_model_path)

    # Freeze most of the layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    #create a new model with the same architecture but allowing fine tuning of the top layers
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    x = base_model.layers[1](inputs) # Skip the input layer and start first hidden layer

    for layer in base_model.layers[2:-2]: # Apply all hidden layers except the output layers
        x = layer(x)
    
    # Policy head
    policy_head = tf.keras.layers.Dense(64, activation='relu')(x)
    policy_output = tf.keras.layers.Dense(action_dim, acticvation='softmax', name='policy')(policy_head)

    # Value head
    value_head = tf.keras.layers.Dense(64, activation='relu')(x)
    value_output = tf.keras.layers.Dense(1, name='value')(value_head)

    # Create model
    model = tf.keras.model.Model(inputs=inputs, outputs=[policy_output, value_output])

    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mean_squared_error'
        }
    )

    return model