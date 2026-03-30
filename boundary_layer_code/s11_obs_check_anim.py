import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import imageio.v2 as imageio  # Use imageio.v2 for compatibility

# Toggle between manual and automatic animation
manual_mode = False 

# Output GIF filename
gif_filename = "animation.gif"
frames = []

# Function to load data from a specified folder
def load_data_from_folder(folder, filename):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        raise FileNotFoundError(f"{filepath} not found")

# Try loading from the model/ folder first
try:
    u_array = load_data_from_folder("model", "u.npy")
    v_array = load_data_from_folder("model", "v.npy")
    pout_array = load_data_from_folder("model", "p_out.npy")
    print("Data loaded from model/ folder.")
except FileNotFoundError:
    try:
        u_array = load_data_from_folder("np_data", "u.npy")
        v_array = load_data_from_folder("np_data", "v.npy")
        pout_array = load_data_from_folder("np_data", "p_out.npy")
        print("Data loaded from np_data/ folder.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

# Normalize u and v
u_array = u_array / 255.0
v_array = v_array / 255.0

# Loop control variable
loop_paused = False

def continue_loop(event):
    global loop_paused
    loop_paused = False

# Create a figure with subplots (3x3)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
plt.subplots_adjust(bottom=0.2)

if manual_mode:
    button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
    button = Button(button_ax, "Continue")
    button.on_clicked(continue_loop)

def highlight(ax=None, **kwargs):
    rect = plt.Rectangle((0, 0), 127, 63, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

n_sim = 1998
for i in range(0, n_sim):
    # Load saved data
    cur_ue = np.load(f"model/obs_data/u_est_{i}.npy")
    cur_ve = np.load(f"model/obs_data/v_est_{i}.npy")
    cur_poute = np.load(f"model/obs_data/cur_poute_{i}.npy")
    cur_ucomp = np.load(f"model/obs_data/cur_ucomp_{i}.npy")
    cur_vcomp = np.load(f"model/obs_data/cur_vcomp_{i}.npy")

    # Clear previous plots
    for ax_row in axes:
        for ax in ax_row:
            ax.clear()

    # First column plots
    axes[0, 0].plot(cur_poute, label="cur_poute")
    axes[0, 0].plot(pout_array[i + 1], label="pout_array[i+1]")
    axes[0, 0].set_title(f"Curves at Step {i + 1}")
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0.,1.5])

    cross_corr = np.corrcoef(cur_poute, pout_array[i + 1])[0, 1]
    cc = 'red' if cross_corr < 0.94 else 'green'
    axes[1, 0].bar(['XCorr'], [cross_corr], color=cc, alpha=0.7)
    axes[1, 0].set_title('Cross-correlation (delay=0)')
    axes[1, 0].set_ylim([-1, 1])
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].grid(True, axis='y', linestyle='--')

    avg_u_squared = np.mean(cur_ucomp ** 2)
    avg_v_squared = np.mean(cur_vcomp ** 2)
    axes[2, 0].bar(['u_comp²', 'v_comp²'], [avg_u_squared, avg_v_squared],
                   color=['blue', 'orange'], alpha=0.7)
    axes[2, 0].set_title('Average of Squared Components')
    axes[2, 0].set_ylim([0, .3])
    axes[2, 0].grid(True, axis='y', linestyle='--')

    # Second column (u-components)
    axes[0, 1].imshow(cur_ue, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title(f"u_est at Step {i + 1}")
    axes[0, 1].axis('off')
    highlight(ax=axes[0, 1], color=cc, linewidth=4)

    axes[1, 1].imshow(u_array[i + 1], cmap="jet", vmin=0, vmax=1)
    axes[1, 1].set_title(f"u_array[i+1] at Step {i + 1}")
    axes[1, 1].axis('off')
    highlight(ax=axes[1, 1], color=cc, linewidth=4)

    axes[2, 1].imshow(cur_ucomp, cmap="jet", vmin=-.1, vmax=.1)
    axes[2, 1].set_title(f"cur_ucomp at Step {i + 1}")
    axes[2, 1].axis('off')

    # Third column (v-components)
    axes[0, 2].imshow(cur_ve, cmap="jet", vmin=0, vmax=1)
    axes[0, 2].set_title(f"v_est at Step {i + 1}")
    axes[0, 2].axis('off')
    highlight(ax=axes[0, 2], color=cc, linewidth=4)

    axes[1, 2].imshow(v_array[i + 1], cmap="jet", vmin=0, vmax=1)
    axes[1, 2].set_title(f"v_array[i+1] at Step {i + 1}")
    axes[1, 2].axis('off')
    highlight(ax=axes[1, 2], color=cc, linewidth=4)

    axes[2, 2].imshow(cur_vcomp, cmap="jet", vmin=-.1, vmax=.1)
    axes[2, 2].set_title(f"cur_vcomp at Step {i + 1}")
    axes[2, 2].axis('off')

    # Draw the plots
    plt.draw()
    fig.canvas.draw()

    # Convert canvas to image and append to frames
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image.copy())

    if manual_mode:
        loop_paused = True
        while loop_paused:
            plt.pause(0.1)
    else:
        plt.pause(0.01)

# Save as GIF
imageio.mimsave(gif_filename, frames[2:], fps=10)
print(f"Saved animation as {gif_filename}")

plt.close()
