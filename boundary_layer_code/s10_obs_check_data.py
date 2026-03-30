import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Create directory to save data
os.makedirs("model/obs_data", exist_ok=True)

# Load data
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")
p_array = np.load("np_data/p.npy")
pout_array = np.load("np_data/p_out.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Load models
sur_model = keras.models.load_model("model/surrogate.h5")
out_model = keras.models.load_model("model/surrogate_p_out.h5")
obs_model = keras.models.load_model("model/observer.h5")

# Simulate observer loop
n_sim = 1998
u_est = [np.random.rand(64, 128)]
v_est = [np.random.rand(64, 128)]

u_est = [u_array[1000]]
v_est = [v_array[1000]]
u_est = [u_array[000]]
v_est = [v_array[000]]

cur_ucomp = np.zeros((64, 128))
cur_vcomp = np.zeros((64, 128))
pout_est = np.zeros((1998,64))

# Loop with percentage progress
for i in range(0,n_sim):
    # Calculate percentage completion
    percent_complete = (i + 1) / n_sim * 100
    print(f"Progress: {percent_complete:.2f}%")

    # Predict p_out
    cur_poute = out_model.predict([
        np.array([u_est[-1]]), 
        np.array([v_est[-1]])
    ], verbose=0) # Suppress TensorFlow output
    cur_poute = cur_poute[0]

    # Predict u and v compensation
    cur_ucomp, cur_vcomp = obs_model.predict([
        np.array([pout_array[i + 1]]), 
        np.array([cur_poute])
    ], verbose=0)
    cur_ucomp = cur_ucomp[0]
    cur_vcomp = cur_vcomp[0]

    # Predict u and v
    cur_ue, cur_ve = sur_model.predict([
        np.array([u_est[-1]]), 
        np.array([v_est[-1]]), 
        np.array([p_array[i]])
    ], verbose=0)



    cur_ue = cur_ue[0] + cur_ucomp
    cur_ve = cur_ve[0] + cur_vcomp
    u_est.append(cur_ue)
    v_est.append(cur_ve)


    # Save data for animation
    np.save(f"model/obs_data/u_est_{i}.npy", cur_ue)
    np.save(f"model/obs_data/v_est_{i}.npy", cur_ve)
    np.save(f"model/obs_data/cur_poute_{i}.npy", cur_poute)
    np.save(f"model/obs_data/cur_ucomp_{i}.npy", cur_ucomp)
    np.save(f"model/obs_data/cur_vcomp_{i}.npy", cur_vcomp)
    pout_est[i,:] = cur_poute

np.savetxt("model/obs_data/pout_est.txt",pout_est)
print("Computation complete. Data saved to model/obs_data/.")