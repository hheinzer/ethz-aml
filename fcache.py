import hashlib
import inspect
import os
import pickle


def fcache(func):
    def wrapper(*args, **kwargs):
        hash_input = inspect.getsource(func).encode()
        hash_input += pickle.dumps(args)
        hash_input += pickle.dumps(sorted(kwargs))
        hash_value = hashlib.sha256(hash_input).hexdigest()
        fname = f"__fcache__/{func.__name__}_{hash_value}.pkl"

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                return pickle.load(f)

        os.makedirs("__fcache__", exist_ok=True)

        output = func(*args, **kwargs)

        with open(fname, "wb") as f:
            pickle.dump(output, f)

        return output

    return wrapper
