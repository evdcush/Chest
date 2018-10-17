######################    Numpy, restore shape      ###########################

def restore_axis_shape(x, ax, d):
    """ Restores an axis ax that was reduced from x
    to it's original shape d
    Just a wrapper for the tedious
    broadcast_to(expand_dims(...)...) op

    Assumes
    -------
        x.ndim >= 1
        ax : int  (only one dim being restored through ax)
    """
    assert x.ndim >= 1 and isinstance(ax, int)

    # Restore shape
    bcast_shape = x.shape[:ax] + (d,) + x.shape[ax:]
    return np.broadcast_to(np.expand_dims(x, ax), bcast_shape)


#######################       Relative imports      ###########################
import os
import sys

# TO GET TRUE FILE DIRECTORY PATH, USE:
os.path.abspath(os.path.dirname(__file__))
#----> EVERYTHING ELSE FUCKED, does pathing from caller path

"""
sandbox
├── data
│   ├── dataset.py
│   └── Iris
│       ├── iris_info.txt
│       ├── iris.npy
│       ├── iris_test.npy
│       └── iris_train.npy
│  
├── deep_learning
│   ├── CrossVal.ipynb
│   ├── functions.py
│   ├── initializers.py
│   ├── layers.py
│   ├── network.py
│   ├── optimizers.py
│   ├── train.py
│   └── utils.py
└── nature
    └── GA.py
"""

# GA <---- dataset
#=======================
sys.insert(1, '..')
from data.dataset import IrisDataset
# OR
fpath = os.path.abspath(os.path.dirname(__file__)) # /home/evan/Projects/AI-Sandbox/sandbox/nature
path_to_dataset = fpath.rstrip(fpath.split('/')[-1]) + 'data'
sys.path.append(path_to_dataset)
