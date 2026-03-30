import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load processed data from the "model/" subfolder
u_next = np.load("model/u_next.npy")
v_next = np.load("model/v_next.npy")
pred_u = np.load("model/pred_u.npy")
pred_v = np.load("model/pred_v.npy")
iter_u = np.load("model/iter_u.npy")
iter_v = np.load("model/iter_v.npy")

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 18))

# Function to update the images in the animation
def update_frame(i):
    if i >= len(u_next): 
        return
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Plot u and v (actual data)
    ax1.imshow(np.concatenate((u_next[i], v_next[i]), axis=1), cmap="jet", vmin=0, vmax=1)
    ax1.set_title(f"Actual: Snapshot {i+1}")
    ax1.axis("off")
    
    # Plot predicted u and v
    ax2.imshow(np.concatenate((pred_u[i], pred_v[i]), axis=1), cmap="jet", vmin=0, vmax=1)
    ax2.set_title(f"Predicted: Snapshot {i+1}")
    ax2.axis("off")
    
    # Plot iterative predicted u and v (starting from u_array[0], v_array[0])
    ax3.imshow(np.concatenate((iter_u[i], iter_v[i]), axis=1), cmap="jet", vmin=0, vmax=1)
    ax3.set_title(f"Iterative Predicted: Snapshot {i+1}")
    ax3.axis("off")

# Create animation
ani = animation.FuncAnimation(
    fig, update_frame, frames=len(u_next), interval=50, repeat=False
)

# Show the animation
plt.tight_layout()
plt.show()