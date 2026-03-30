import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load input files
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")
p_array = np.load("np_data/p.npy")

# Load output file (target values)
p_out_array = np.load("np_data/p_out.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

input_shape_p = (p_array.shape[1],)
output_shape_p = (p_out_array.shape[1],)

# Split data
ids = list(range(u_array.shape[0]))
train_size = 1500
X_train_u, X_val_u = u_array[ids][:train_size], u_array[ids][train_size:]
X_train_v, X_val_v = v_array[ids][:train_size], v_array[ids][train_size:]

X_train_u = X_train_u[..., None]
X_train_v = X_train_v[..., None]

y_train_p_out, y_val_p_out = p_out_array[ids][:train_size], p_out_array[ids][train_size:]

# Define CNN Model
input_u = keras.Input(shape=(64, 128, 1), name="input_u")
input_v = keras.Input(shape=(64, 128, 1), name="input_v")

merged_states = layers.Concatenate(axis = -1)([input_u,input_v])
l2_lambda = 0.000001

# CNN layers
conv_layers = [
    layers.Conv2D(4, (3, 3), activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(2, (3, 3), activation="linear", padding="same", kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.MaxPooling2D((2, 2)),
]


features = merged_states
for layer in conv_layers:
    features = layer(features)

out1 = layers.Lambda(lambda x: tf.gather(x, 0, axis=-1))(features)
out2 = layers.Lambda(lambda x: tf.gather(x, 1, axis=-1))(features)

out1 = layers.Reshape((32,))(out1)
out2 = layers.Reshape((32,))(out2)

# Output layer for predicting p_out
output_p_out = layers.Concatenate(axis=1)((out1,out2))

# Calculate steps per epoch
batch_size = 128
steps_per_epoch = len(X_train_u) // batch_size
decay_steps = 100 * steps_per_epoch

# Define learning rate schedule
initial_learning_rate = 0.0015
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=0.93,
    staircase=True
)

# Use this in your optimizer
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model = keras.Model(inputs=[input_u, input_v], outputs=output_p_out)
model.compile(optimizer=optimizer, loss="mse")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Custom callback to plot loss every 10 epochs
class LossPlotCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:  # Every 10 epochs
            plt.clf()
            plt.plot(range(1, epoch + 1), self.model.history.history['loss'], label='Training Loss')
            plt.plot(range(1, epoch + 1), self.model.history.history['val_loss'], label='Validation Loss')
            plt.title('Loss Function during Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (Log Scale)')
            plt.legend()
            plt.yscale('log')  # Set y-axis to log scale
            plt.grid(True)
            plt.pause(0.1)  # Pause to update plot


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current step from the optimizer
        step = self.model.optimizer.iterations.numpy()
        # Compute the current learning rate using the schedule
        lr = self.model.optimizer.learning_rate(step).numpy()
        print(f"Epoch {epoch + 1}: Learning rate = {lr:.6f}")

# Train the model with callback
history = model.fit(
    [X_train_u, X_train_v],  # Inputs
    y_train_p_out,  # Target
    epochs=1500,  # Number of epochs
    batch_size=batch_size,  # Batch size
    validation_data=([X_val_u, X_val_v], y_val_p_out),
    callbacks=[LossPlotCallback(), LearningRateLogger()],  # Plotting callback
    shuffle=True
)

# Save the trained model
model.save("model/surrogate_p_out.h5")
