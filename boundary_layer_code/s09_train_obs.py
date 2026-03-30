import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Load input files
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")
p_array = np.load("np_data/p.npy")
pout_array = np.load("np_data/p_out.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Compensation and initial u and v data
ucomp_array = np.zeros(u_array.shape)
vcomp_array = np.zeros(v_array.shape)
idse = list(range(u_array.shape[0]))
n=20
spread = np.random.randint(low=-n, high=n+1, size=u_array.shape[0])
idse = idse+spread
idse = np.clip(idse, 0, u_array.shape[0]-1)
ue_array = u_array[idse]
ve_array = v_array[idse]

# Dimensions    
p_size = p_array.shape[1]

# Hyperparameters
nh = 8
d_size = len(u_array)
train_size = int(0.95*d_size) 
train_size = 1500
lrate = 1e-3
n_warm = 100
n_epochs = 600

# Train/validation Ids
train_ids = list(range(train_size))[:-nh]
valid_ids = list(range(train_size,d_size))[:-nh]

# Train/validation data
u_train = u_array[train_ids]
v_train = v_array[train_ids]
ucomp_train = ucomp_array[train_ids]
vcomp_train = vcomp_array[train_ids]
ue_train = ue_array[train_ids]
ve_train = ve_array[train_ids]

u_valid = u_array[valid_ids]
v_valid = v_array[valid_ids]
ucomp_valid = ucomp_array[valid_ids]
vcomp_valid = vcomp_array[valid_ids]
ue_valid = ue_array[valid_ids]
ve_valid = ve_array[valid_ids]

p_train = []
p_valid = []
for step in range(nh):
    p_train.append(p_array[np.array(train_ids)+step])
    p_valid.append(p_array[np.array(valid_ids)+step])



# Models
sur_model = keras.models.load_model('model/surrogate.h5')
out_model = keras.models.load_model('model/surrogate_p_out.h5')

# Observer model
input_obs1 = keras.Input(shape=(pout_array.shape[1],), name="input_obs1")
input_obs2 = keras.Input(shape=(pout_array.shape[1],), name="input_obs2")

u_feats1 = layers.Lambda(lambda x: x[:,:32])(input_obs1)
v_feats1 = layers.Lambda(lambda x: x[:,32:])(input_obs1)

u_feats2 = layers.Lambda(lambda x: x[:,:32])(input_obs2)
v_feats2 = layers.Lambda(lambda x: x[:,32:])(input_obs2)

feats1 = layers.Concatenate(axis=3)((layers.Reshape((4,8,1))(u_feats1),layers.Reshape((4,8,1))(v_feats1)))
feats2 = layers.Concatenate(axis=3)((layers.Reshape((4,8,1))(u_feats2),layers.Reshape((4,8,1))(v_feats2)))

merged = layers.Concatenate(axis=-1)([feats1,feats2])

l2_lambda = 0.0001
nno_layers=[
    layers.Conv2DTranspose(16,kernel_size=(3,3),strides=(2, 2),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2D(16,kernel_size=(3,3),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2DTranspose(8 ,kernel_size=(3,3),strides=(2, 2),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2D(8,kernel_size=(3,3),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2DTranspose(4 ,kernel_size=(3,3),strides=(2, 2),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2D(4,kernel_size=(3,3),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2DTranspose(2 ,kernel_size=(3,3),strides=(2, 2),padding='same',activation='relu', kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.Conv2D(2,kernel_size=(3,3),padding='same',activation='linear', kernel_regularizer=keras.regularizers.l2(l2_lambda))
]

obs_features = merged

for layer in nno_layers:
    obs_features = layer(obs_features)

obscomp_u = layers.Lambda(lambda x: x[:, :, :, 0:1])(obs_features)
obscomp_v = layers.Lambda(lambda x: x[:, :, :, 1:2])(obs_features)
obs_model = keras.Model(inputs=[input_obs1, input_obs2], outputs=[obscomp_u, obscomp_v])

# Get trainable weights
trainable_vars = []
for layer in nno_layers:
    weights = layer.weights
    for w in weights:
        trainable_vars.append(w)

# Closed-loop model inputs
input_u = keras.Input(shape=(64, 128), name="input_u")
input_v = keras.Input(shape=(64, 128), name="input_v")
inputs_p = [keras.Input([p_size]) for _ in range(nh)]
inputs_noise = [keras.Input([64]) for _ in range(nh)]

input_ue = keras.Input(shape=(64, 128), name="input_ue")
input_ve = keras.Input(shape=(64, 128), name="input_ve")

# List of tensors to compute errors
error_list = []
u_error_list = []
v_error_list = []
u_comp_list = []
v_comp_list = []

# Closed-loop stuff
cur_u = input_u
cur_v = input_v
cur_ue = input_ue
cur_ve = input_ve

deb_ur = []
deb_vr = []
deb_ue = []
deb_ve = []
deb_poute = []
deb_poutr = []
for i in range(nh):
    cur_p = inputs_p[i]
    cur_noise = inputs_noise[i]

    cur_pout = out_model([cur_u,cur_v])
    cur_u, cur_v = sur_model([cur_u,cur_v,cur_p])
    deb_ur.append(cur_u)
    deb_vr.append(cur_v)
    deb_poutr.append(cur_pout)

    cur_poute = out_model([cur_ue,cur_ve])
    cur_ue, cur_ve = sur_model([cur_ue,cur_ve,cur_p])

    cur_ucomp, cur_vcomp = obs_model([cur_pout+cur_noise,cur_poute])
    u_comp_list.append(cur_ucomp)
    v_comp_list.append(cur_vcomp)

    cur_ue = layers.Add()([cur_ue,cur_ucomp])
    cur_ve = layers.Add()([cur_ve,cur_vcomp])

    deb_ue.append(cur_ue)
    deb_ve.append(cur_ve)
    deb_poute.append(cur_poute)

    error_list.append(cur_pout - cur_poute)
    u_error_list.append(cur_u-cur_ue)
    v_error_list.append(cur_v-cur_ve)



# Loss function
wu = 1.
wv = 1.
wucomp = 0.1
wvcomp = 0.1

pout_loss = tf.square(tf.stack(error_list))
pout_loss = tf.reduce_mean(pout_loss, axis=-1)
pout_loss = tf.reduce_mean(pout_loss, axis=-1)
pout_loss = tf.reduce_mean(pout_loss, axis=-1)

u_loss = tf.square(tf.stack(u_error_list))
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)
u_loss = tf.reduce_mean(u_loss, axis=-1)

v_loss = tf.square(tf.stack(v_error_list))
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)
v_loss = tf.reduce_mean(v_loss, axis=-1)

ucomp_loss = tf.square(tf.stack(u_comp_list))
ucomp_loss = tf.reduce_mean(ucomp_loss, axis=-1)
ucomp_loss = tf.reduce_mean(ucomp_loss, axis=-1)
ucomp_loss = tf.reduce_mean(ucomp_loss, axis=-1)
ucomp_loss = tf.reduce_mean(ucomp_loss, axis=-1)
ucomp_loss = tf.reduce_mean(ucomp_loss, axis=-1)

vcomp_loss = tf.square(tf.stack(v_comp_list))
vcomp_loss = tf.reduce_mean(vcomp_loss, axis=-1)
vcomp_loss = tf.reduce_mean(vcomp_loss, axis=-1)
vcomp_loss = tf.reduce_mean(vcomp_loss, axis=-1)
vcomp_loss = tf.reduce_mean(vcomp_loss, axis=-1)
vcomp_loss = tf.reduce_mean(vcomp_loss, axis=-1)

loss = pout_loss + wu*u_loss + wv*v_loss + wucomp*ucomp_loss + wv*vcomp_loss

tfLRate = tf.Variable(initial_value=0, trainable=False, dtype=tf.float32)
adamOptimizer = tf.train.AdamOptimizer(learning_rate=tfLRate)
opt = adamOptimizer.minimize(loss, var_list=trainable_vars)

# Training data
trainFd = {
        input_u: u_train,
        input_v: v_train,
        input_ue: ue_train,
        input_ve: ve_train}
trainFd.update({inputs_p[k]: p_train[k] for k in range(len(p_train))})
stdn = 0.18
trainFd.update({inputs_noise[k]: np.random.normal(0, stdn, [u_train.shape[0],64]) for k in range(len(p_train))})


# Validation data
validFd = {
        input_u: u_valid,
        input_v: v_valid,
        input_ue: ue_valid,
        input_ve: ve_valid}
validFd.update({inputs_p[k]: p_valid[k] for k in range(len(p_valid))})
validFd.update({inputs_noise[k]: np.random.normal(0, stdn, [u_valid.shape[0],64]) for k in range(len(p_train))})

# Batch selection function
def getBatch(dataDic,position):
        dataSize = dataDic[input_u].shape[0]

        start = position*batch_size
        end = min((position+1)*batch_size, dataSize)

        batch = {
                input_u: dataDic[input_u][start:end],
                input_v: dataDic[input_v][start:end],
                input_ue: dataDic[input_ue][start:end],
                input_ve: dataDic[input_ve][start:end]
                }
        batch.update({inputs_p[k]: dataDic[inputs_p[k]][start:end] for k in range(len(inputs_p))})
        batch.update({inputs_noise[k]: dataDic[inputs_noise[k]][start:end] for k in range(len(inputs_noise))})
        
        return batch

# Initializing variables while backing up trained weights
sess = tf.compat.v1.keras.backend.get_session()
w_sur = sur_model.get_weights()
w_out = out_model.get_weights()
sess.run(tf.global_variables_initializer())
sur_model.set_weights(w_sur)
out_model.set_weights(w_out)

# Initialize plot for training/validation loss
plt.ion() 
fig, ax = plt.subplots()
train_loss_line, = ax.plot([], [], label="trainLoss", color='b')
valid_loss_line, = ax.plot([], [], label="validLoss", color='r')
plt.yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()

# Training
batch_size = 64
printStr1 = "Epoch {}/{} - TrainLoss: {:.4e} | ValidLoss: {:.4e} (pout: {:.4e}, u: {:.4e}, v: {:.4e}, uc: {:.4e}, vc: {:.4e})"
force_stop = False

with sess.as_default():
    for i in range(n_epochs):
        if i < n_warm:
             cur_lrate = lrate*i/n_warm
        else:
             cur_lrate = lrate
        tf.keras.backend.set_value(tfLRate,cur_lrate)

        its = range(int(train_size/batch_size)+1)
        n_its = len(its)
        trainLoss = 0
        for j in its:

            cur_batch = getBatch(trainFd,j)
            opt.run(feed_dict=cur_batch) 
        
            trainLoss = trainLoss + sess.run(loss, feed_dict=cur_batch)/n_its

        valid_total, valid_pout, valid_u, valid_v, valid_uc, valid_vc = sess.run(
            [loss, pout_loss, u_loss, v_loss, ucomp_loss, vcomp_loss], feed_dict=validFd)

        # Update plot with new loss values
        train_loss_line.set_xdata(np.append(train_loss_line.get_xdata(), i+1))
        train_loss_line.set_ydata(np.append(train_loss_line.get_ydata(), trainLoss))
        valid_loss_line.set_xdata(np.append(valid_loss_line.get_xdata(), i+1))
        valid_loss_line.set_ydata(np.append(valid_loss_line.get_ydata(), valid_total))

        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        print(printStr1.format(i+1, n_epochs, trainLoss, valid_total, valid_pout, valid_u, valid_v, valid_uc, valid_vc, end='\r'))

        if force_stop:
            break


                


obs_model.save("model/observer.h5")

