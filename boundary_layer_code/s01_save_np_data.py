import numpy as np
import glob
from PIL import Image

# Horizontal components
file_pattern = "training_data/u*.png"
image_paths = sorted(glob.glob(file_pattern)) 

images = [np.array(Image.open(img)) for img in image_paths]
u_array = np.array(images)

np.save("np_data/u.npy", u_array)

print(f"Saved U data to np_data/u.npy")

# Vertical components
file_pattern = "training_data/v*.png"
image_paths = sorted(glob.glob(file_pattern)) 

images = [np.array(Image.open(img)) for img in image_paths]
v_array = np.array(images)

np.save("np_data/v.npy", v_array)

print(f"Saved V data to np_data/v.npy")

# Low-resolution data and boundary conditions
p = np.loadtxt('training_data/p.txt')
p = p[:,21:]
np.save('np_data/p.npy',p)
print(f"Saved boundary conditions to np_data/p.npy")

p_out1 = np.reshape(u_array[:,7::16,7::16],(1999,32))/255.
p_out2 = np.reshape(v_array[:,7::16,7::16],(1999,32))/255.
p_out = np.concatenate((p_out1, p_out2),axis=1)

np.save('np_data/p_out.npy',p_out)
print(f"Saved sensor data to np_data/p_out.npy")