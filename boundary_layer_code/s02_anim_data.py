import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load data
u_images = np.load("np_data/u.npy") 
v_images = np.load("np_data/v.npy")  
p_curves = np.load("np_data/p.npy") 

n_samples, n_points = p_curves.shape 

# Create figure and axes
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  
ax_u, ax_v, ax_p = axes

# Display first frame images
im_u = ax_u.imshow(u_images[0], cmap="gray")
ax_u.set_title("U Images")
ax_u.axis("off")

im_v = ax_v.imshow(v_images[0], cmap="gray")
ax_v.set_title("V Images")
ax_v.axis("off")

# Sensor data curve setup
ax_p.set_title("Sensor data curves")
ax_p.set_xlim(0, n_points - 1) 
ax_p.set_ylim(np.min(p_curves), np.max(p_curves)) 
curve_plot, = ax_p.plot([], [], 'r-', lw=2)

# Animation function
def update(frame):
    im_u.set_data(u_images[frame])
    im_v.set_data(v_images[frame])
    curve_plot.set_data(np.arange(n_points), p_curves[frame])
    return im_u, im_v, curve_plot

# Run animation
ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=50, blit=False)

plt.tight_layout()
plt.show()