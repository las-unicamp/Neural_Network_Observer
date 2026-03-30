import os
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.disable_v2_behavior()
# For TF 2.4.0, use this approach
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


# Custom callback to plot loss every n epochs
class PlotLossCallback(keras.callbacks.Callback):
    def __init__(self, n):
        super(PlotLossCallback, self).__init__()
        self.n = n
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def on_epoch_end(self, epoch, logs=None):
        # Append loss values
        self.epochs.append(epoch)
        self.train_loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])

        # Plot every n epochs
        if (epoch + 1) % self.n == 0 or epoch == 0:
            self.ax.clear()
            self.ax.plot(self.epochs, self.train_loss, label="Training Loss")
            self.ax.plot(self.epochs, self.val_loss, label="Validation Loss")
            self.ax.set_title(f"Loss Function at Epoch {epoch + 1}")
            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel("Loss (Log Scale)")
            self.ax.legend()
            self.ax.set_yscale("log")
            self.ax.grid(True)
            plt.draw()
            plt.pause(0.1)


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        step = self.model.optimizer.iterations.numpy()
        lr = self.model.optimizer.learning_rate(step).numpy()
        print(f"Epoch {epoch + 1}: Learning rate = {lr:.6f}")


# Add the callback to training
learning_rate_logger = LearningRateLogger()
nh = 8

# Input shapes for u, v and boundary conditions p
input_shape_uv = (64, 128, 1)
input_shape_p = np.load("np_data/p.npy").shape[1]
n_data = np.load("np_data/p.npy").shape[0]

input_p = keras.Input(shape=input_shape_p, name="input_p")

# U and V inputs
input_u = keras.Input(shape=input_shape_uv, name="input_u")
input_v = keras.Input(shape=input_shape_uv, name="input_v")

merged_states = layers.Concatenate(axis=-1)([input_u, input_v])

bound_u = layers.Lambda(lambda x: tf.reverse(x[:, 0:32], axis=[1]))(input_p)
bound_u = layers.Reshape([32, 1])(bound_u)
bound_u = layers.UpSampling1D(size=2)(bound_u)
bound_u = layers.Reshape([64, 1, 1])(bound_u)

bound_v = layers.Lambda(lambda x: tf.reverse(x[:, 32:64], axis=[1]))(input_p)
bound_v = layers.Reshape([32, 1])(bound_v)
bound_v = layers.UpSampling1D(size=2)(bound_v)
bound_v = layers.Reshape([64, 1, 1])(bound_v)


merged_bound = keras.layers.Concatenate(axis=-1)([bound_u, bound_v])

merged = keras.layers.Concatenate(axis=2)([merged_bound, merged_states])

# Some hyperparameters
l2_lambda = 0.00
batch_size = 64

feats = merged
feats = layers.Conv2D(
    16,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=keras.regularizers.l2(l2_lambda),
)(feats)
feats = layers.Conv2D(
    16,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=keras.regularizers.l2(l2_lambda),
)(feats)
feats = layers.Conv2D(
    2,
    (3, 3),
    activation="linear",
    padding="same",
    kernel_regularizer=keras.regularizers.l2(l2_lambda),
)(feats)
feats = layers.Lambda(lambda x: x[:, :, :-1, :])(feats)

# Output layers for next timestep u and v using deconvolution
output_u = layers.Lambda(lambda x: x[..., 0:1])(feats)
output_v = layers.Lambda(lambda x: x[..., 1:2])(feats)

# Define the model
model = keras.Model(inputs=[input_u, input_v, input_p], outputs=[output_u, output_v])

# Define the initial learning rate and decay rate
initial_learning_rate = 0.001
decay_rate = 0.9  # Decay rate per step
decay_steps = 50 * int(math.ceil(n_data / batch_size))

# Create the learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True,
)

# Compile the model with the optimizer using the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mse")

# Load data
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")
p_array = np.load("np_data/p.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Create targets (next timestep values)
u_next = u_array[1:]
v_next = v_array[1:]

# Remove the last sample for input-output alignment
u_array = u_array[:-1]
v_array = v_array[:-1]
p_array = p_array[:-1]

# Custom callback
plot_loss_callback = PlotLossCallback(n=10)  # Plot every 10 epochs

rec_input_u = keras.Input(shape=input_shape_uv, name="rec_input_u")
rec_input_v = keras.Input(shape=input_shape_uv, name="rec_input_v")
rec_inputs_p = [
    keras.Input(shape=input_shape_p, name=f"rec_input_p_{i}") for i in range(nh)
]

labels_u = [keras.Input(shape=input_shape_uv, name=f"label_u_{i}") for i in range(nh)]
labels_v = [keras.Input(shape=input_shape_uv, name=f"label_v_{i}") for i in range(nh)]

rec_outputs_u = []
rec_outputs_v = []

cur_input_u = rec_input_u
cur_input_v = rec_input_v

error_u = []
error_v = []
for i in range(nh):
    cur_input_p = rec_inputs_p[i]
    cur_outs = model([cur_input_u, cur_input_v, cur_input_p])
    cur_u_out = cur_outs[0]
    cur_v_out = cur_outs[1]

    rec_outputs_u.append(cur_u_out)
    rec_outputs_v.append(cur_v_out)

    cur_input_u = cur_u_out
    cur_input_v = cur_v_out

    error_u.append(cur_u_out - labels_u[i])
    error_v.append(cur_v_out - labels_v[i])

u_loss = tf.square(tf.stack(error_u))
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)

v_loss = tf.square(tf.stack(error_v))
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)

loss = u_loss + v_loss

ns = len(u_array)

u_array = np.expand_dims(u_array, axis=-1)
v_array = np.expand_dims(v_array, axis=-1)

rec_u_data_in = u_array[:-nh]
rec_v_data_in = v_array[:-nh]
rec_p_data_in = [p_array[i : ns - nh + i] for i in range(nh)]

rec_u_data_labels = [u_array[i + 1 : ns - nh + i + 1] for i in range(nh)]
rec_v_data_labels = [v_array[i + 1 : ns - nh + i + 1] for i in range(nh)]

train_size = 1500

rec_u_data_in_train = rec_u_data_in[:train_size]
rec_v_data_in_train = rec_v_data_in[:train_size]
rec_p_data_in_train = [data[:train_size] for data in rec_p_data_in]
rec_u_data_labels_train = [data[:train_size] for data in rec_u_data_labels]
rec_v_data_labels_train = [data[:train_size] for data in rec_v_data_labels]

rec_u_data_in_valid = rec_u_data_in[train_size:]
rec_v_data_in_valid = rec_v_data_in[train_size:]
rec_p_data_in_valid = [data[train_size:] for data in rec_p_data_in]
rec_u_data_labels_valid = [data[train_size:] for data in rec_u_data_labels]
rec_v_data_labels_valid = [data[train_size:] for data in rec_v_data_labels]

# Training data
train_fd = {rec_input_u: rec_u_data_in_train, rec_input_v: rec_v_data_in_train}
train_fd.update({rec_inputs_p[k]: rec_p_data_in_train[k] for k in range(nh)})
train_fd.update({labels_u[k]: rec_u_data_labels_train[k] for k in range(nh)})
train_fd.update({labels_v[k]: rec_v_data_labels_train[k] for k in range(nh)})

# Validation data
valid_fd = {rec_input_u: rec_u_data_in_valid, rec_input_v: rec_v_data_in_valid}
valid_fd.update({rec_inputs_p[k]: rec_p_data_in_valid[k] for k in range(nh)})
valid_fd.update({labels_u[k]: rec_u_data_labels_valid[k] for k in range(nh)})
valid_fd.update({labels_v[k]: rec_v_data_labels_valid[k] for k in range(nh)})


# Batch selection function
def getBatch(dataDic, position):
    dataSize = dataDic[rec_input_u].shape[0]

    start = position * batch_size
    end = min((position + 1) * batch_size, dataSize)

    batch = {
        rec_input_u: dataDic[rec_input_u][start:end],
        rec_input_v: dataDic[rec_input_v][start:end],
    }
    batch.update(
        {
            rec_inputs_p[k]: dataDic[rec_inputs_p[k]][start:end]
            for k in range(len(rec_inputs_p))
        }
    )
    batch.update(
        {labels_u[k]: dataDic[labels_u[k]][start:end] for k in range(len(labels_u))}
    )
    batch.update(
        {labels_v[k]: dataDic[labels_v[k]][start:end] for k in range(len(labels_v))}
    )

    return batch


n_epochs = 1000
n_warm = 50
lrate = 0.001
tfLRate = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
adam_optimizer = tf.train.AdamOptimizer(learning_rate=tfLRate)
opt = adam_optimizer.minimize(loss)

# sess = tf.compat.v1.keras.backend.get_session()
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

sess.run(tf.global_variables_initializer())

# Initialize plot for training/validation loss
plt.ion()
fig, ax = plt.subplots()
(train_loss_line,) = ax.plot([], [], label="trainLoss", color="b")
(valid_loss_line,) = ax.plot([], [], label="validLoss", color="r")
plt.yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

force_stop = False

printStr1 = "Epoch {}/{} - TrainLoss: {:.4e} | ValidLoss: {:.4e}"
with sess.as_default():
    for i in range(n_epochs):
        if i < n_warm:
            cur_lrate = lrate * i / n_warm
        else:
            cur_lrate = lrate
        tf.keras.backend.set_value(tfLRate, cur_lrate)

        its = range(int(train_size / batch_size) + 1)
        n_its = len(its)
        train_loss = 0
        for j in its:

            cur_batch = getBatch(train_fd, j)
            opt.run(feed_dict=cur_batch)

            train_loss = train_loss + sess.run(loss, feed_dict=cur_batch) / n_its

        (valid_total,) = sess.run([loss], feed_dict=valid_fd)

        # Update plot with new loss values
        train_loss_line.set_xdata(np.append(train_loss_line.get_xdata(), i + 1))
        train_loss_line.set_ydata(np.append(train_loss_line.get_ydata(), train_loss))
        valid_loss_line.set_xdata(np.append(valid_loss_line.get_xdata(), i + 1))
        valid_loss_line.set_ydata(np.append(valid_loss_line.get_ydata(), valid_total))

        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        print(printStr1.format(i + 1, n_epochs, train_loss, valid_total, end="\r"))

        if force_stop:
            break

# Save the model
os.makedirs("model", exist_ok=True)
model.save("model/surrogate.h5")
