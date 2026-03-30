# Instructions

Create your python environment (working with python 38.8.10):
    python3 -m venv virtual_env
    source virtual_env/bin/activate
    pip install -r requirements.txt

Run the following scripts:

s01_train_model_and_control.py
s02_train_observer.py

# s01_train_model_and_control.py

To change parameters, edit parameters.py.

If you are running it for the first time, set the parameter parData['loadRestart'] to False.

This routine will train and retrain the NNSM and NNC along a number of iterations provided in parGeneral['nIterations'].

The trained networks are saved to output/ every iteration. 

To restart from a specific saved iteration n, set parData['loadRestart']=True and parData['restartIndex']=n.

# s02_train_observer.py

Set parData['loadRestart'] = True and choose which NNSM model will be used via parData['restartIndex']=n. The model trained in the n-th iteration through the previous script will be loaded.

