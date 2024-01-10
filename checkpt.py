import os
import pickle


def checkpoint(name, func, *args, **kwargs):
    fname = "__checkpt__/" + name + ".pkl"
    try:
        with open(fname, "rb") as f:
            return pickle.load(f)
    except:
        res = func(*args, **kwargs)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "wb") as f:
            pickle.dump(res, f)
        return res
