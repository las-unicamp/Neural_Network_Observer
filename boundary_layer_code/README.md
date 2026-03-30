# Instructions

Build the docker image:

docker build -t bl_nno:latest .

Run the container 

docker run --rm -it --network host --env DISPLAY=$DISPLAY -v $PWD:/workspace --gpus all --ipc=host --name nno --shm-size=4g bl_nno:latest

Within the container, at /workspace directory, run the scripts in order: s01_..., s02_..., ...


# s01_save_np_data.py

Saves data (states and boundary conditions) in numpy format.

# s02_anim_data.py

Visualize data:

1st column: vertical velocity component;

2nd column: horizontal velocity component;

3rd column: boundary conditions (first half is U data and second half is V data).

# s03_conv_train.py

Train NNSM from data.

# s04_training_check_data.py

Save NNSM simulation for visualization.

# s05_training_check_anim.py

Visualize NNSM results through direct inference from data at previous time step and through autoregressive updates from initial conditions.



# s06_train_out_model.py

Train output model to map states to sensor outputs. Direct slicing is avoided since we do not assume a priori knowledge of the model.

Giving NaN error

# s07_out_check_data.py

Save output model inference for visualization.

# s08_out_check_anim.py

Create video for vizualization of the model results.

# s09_train_obs.py

Train the NN observer in closed loop.

# s10_obs_check_data.py

Save data for vizualiation of estimations.

# s11_obs_check_anim.py

Visualize observer results
