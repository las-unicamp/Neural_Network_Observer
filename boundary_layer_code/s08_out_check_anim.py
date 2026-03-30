import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Starting sample
st = 1400

# Load p_out (ground truth) and p_out_pred (predicted)
p_out = np.load("np_data/p_out.npy")[st:]
p_out_pred = np.load("model/p_out_pred.npy")[st:]

# Ensure both have the same number of samples
if p_out.shape != p_out_pred.shape:
    raise ValueError(f"Mismatch in shapes: p_out {p_out.shape}, p_out_pred {p_out_pred.shape}")

n_samples, n_points = p_out.shape

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Ground Truth vs Prediction")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_xlim(0, n_points)
ax.set_ylim(min(p_out.min(), p_out_pred.min()), max(p_out.max(), p_out_pred.max()))
ax.grid(True)

# Initialize line plots
line_gt, = ax.plot([], [], "b-", label="Ground Truth")
line_pred, = ax.plot([], [], "r-", label="Prediction")
ax.legend()

# Animation update function
def update(frame):
    line_gt.set_data(np.arange(n_points), p_out[frame])
    line_pred.set_data(np.arange(n_points), p_out_pred[frame])
    ax.set_title(f"Sample {frame+1}/{n_samples}")
    return line_gt, line_pred

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=100, blit=False, repeat=False)


plt.show()