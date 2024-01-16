import gzip
import pickle


def load_pkl(fname):
    with gzip.open(fname, "rb") as file:
        return pickle.load(file)  # type: ignore


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as file:
        pickle.dump(data, file, 2)  # type: ignore
