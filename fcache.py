import hashlib
import inspect
import os
import pickle


def fcache(func):
    def wrapper(*args, **kwargs):
        hash = inspect.getsource(func).encode()
        hash += pickle.dumps(args)
        hash += pickle.dumps(sorted(kwargs))
        hash = hashlib.sha256(hash).hexdigest()
        fname = f"__fcache__/{func.__name__}_{hash}.pkl"

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                return pickle.load(f)

        if not os.path.exists("__fcache__"):
            os.mkdir("__fcache__")

        output = func(*args, **kwargs)

        with open(fname, "wb") as f:
            pickle.dump(output, f)

        return output

    return wrapper
