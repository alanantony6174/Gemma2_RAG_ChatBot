import os
import pickle

def initialize_pickle_file():
    if THREAD_HISTORY_PICKLE_PATH and not os.path.exists(THREAD_HISTORY_PICKLE_PATH):
        with open(THREAD_HISTORY_PICKLE_PATH, "wb") as out_file:
            pickle.dump([], out_file)

initialize_pickle_file()
