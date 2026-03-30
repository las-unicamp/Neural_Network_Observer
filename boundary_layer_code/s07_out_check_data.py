import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load input files
u_array = np.load("np_data/u.npy")
v_array = np.load("np_data/v.npy")

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Load trained model
model_path = "model/surrogate_p_out.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
model = keras.models.load_model(model_path)

# Predict p_out
p_out_pred = model.predict([u_array, v_array])


# Save the predicted p_out
os.makedirs("model", exist_ok=True)
np.save("model/p_out_pred.npy", p_out_pred)

print(f"Predicted p_out saved to 'model/p_out_pred.npy'. Shape: {p_out_pred.shape}")