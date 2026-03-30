import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Load data
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")
p_array = np.load("np_data/p.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Load the saved model
model = keras.models.load_model("model/surrogate.h5")

# Create targets (next timestep values)
u_next = u_array[1:]  
v_next = v_array[1:]  

# Remove the last sample for input-output alignment
u_array = u_array[:-1]
v_array = v_array[:-1]
p_array = p_array[:-1]

# Apply the model to the data
pred_u, pred_v = model.predict([u_array, v_array, p_array])
pred_u = pred_u[:-1]
pred_v = pred_v[:-1]

# Iterative prediction
iter_u = [u_array[0]]
iter_v = [v_array[0]]

# Loop to generate the predictions iteratively
nsim = 1000
for i in range(nsim):
    # Predict the next u and v using the model
    next_u, next_v = model.predict([np.array([iter_u[-1]]), np.array([iter_v[-1]]), np.array([p_array[i]])])
    
    # Append the predicted u and v to the lists
    iter_u.append(next_u[0])
    iter_v.append(next_v[0])
    
    # Print progress every 10 iterations
    if i % 10 == 0 or i == nsim - 1: 
        progress = (i + 1) / nsim * 100
        print(f"Progress: {progress:.2f}%", end="\r")

# Remove the initial condition from iterative predictions
iter_u = iter_u[1:]
iter_v = iter_v[1:]

# Save processed data to the "model/" subfolder
os.makedirs("model", exist_ok=True)  
np.save("model/u_next.npy", u_next)
np.save("model/v_next.npy", v_next)
np.save("model/pred_u.npy", pred_u)
np.save("model/pred_v.npy", pred_v)
np.save("model/iter_u.npy", np.array(iter_u))
np.save("model/iter_v.npy", np.array(iter_v))

print("Data processing complete. Results saved to 'model/'.")