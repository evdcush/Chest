
from pprint import pprint
import yaml

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# yaml rw helpers
# ===============
def R_yml(fname):
    with open(fname) as file:
        return yaml.load(file)

def W_yml(fname, obj):
    with open(fname, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)


# Experiment
# ==========
yml = R_yml('mess.yml')

# play with embedded code


def main():
    print('yaml file var name: yml')

if __name__ == '__main__':
    main()
