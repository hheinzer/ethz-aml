import os
import pickle


def checkpoint(name, func, args):
    try:
        return load_checkpoint(name)
    except:
        res = func(*args)
        save_checkpoint(name, tuple(res))
        return res


def load_checkpoint(name):
    with open("__checkpt__/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def save_checkpoint(name, args):
    os.makedirs("__checkpt__", exist_ok=True)
    with open("__checkpt__/" + name + ".pkl", "wb") as f:
        pickle.dump(args, f)
