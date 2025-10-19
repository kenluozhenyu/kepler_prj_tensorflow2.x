# Utility to write/read the [global view, local_view] data to/from files.
#
# The write function is to be used to generate training records (including
# train/validation/test sets.
#
# The read function is to be used when doing training.
# Corresponding generators will use this function to get the training data
# to the CNN model.

import numpy as np

import os.path
import pickle

from data import preprocess
from configure import environment


def tce_global_view_local_view_to_file(tce_data):
    X1, X2 = [], []

    kep_id = "%.9d" % int(tce_data.kepid)

    kepid_dir = os.path.join(environment.KEPLER_DATA_FOLDER, kep_id[0:4], kep_id)
    file_name = os.path.join(kepid_dir, "{0:09d}_plnt_num-{1:02d}_tce.record".format(int(tce_data.kepid), tce_data.tce_plnt_num))

    if not os.path.isfile(file_name):
        time, flux = preprocess.read_and_process_light_curve(tce_data.kepid, environment.KEPLER_DATA_FOLDER, 0.75)
        time, flux = preprocess.phase_fold_and_sort_light_curve(
            time, flux, tce_data.tce_period, tce_data.tce_time0bk)

        global_view = preprocess.global_view(time, flux, tce_data.tce_period)
        local_view = preprocess.local_view(time, flux, tce_data.tce_period, tce_data.tce_duration)

        # Reshape for the Keras CNN model input
        global_view = np.reshape(global_view, (2001, 1))
        local_view = np.reshape(local_view, (201, 1))

        X1.append(global_view)
        X2.append(local_view)

        result_X = [np.array(X1), np.array(X2)]

        with open(file_name, 'wb') as fp:
            pickle.dump(result_X, fp)
        fp.close()

    else:
        print("Global view file and local view file already exist. Skipped.\n")


# This function will get the file location from the tce.kepid, and then
# get the '.record' file and read the value
def read_tce_global_view_local_view_from_tce(tce_data):
    result_X = []

    kep_id = "%.9d" % int(tce_data.kepid)

    kepid_dir = os.path.join(environment.KEPLER_DATA_FOLDER, kep_id[0:4], kep_id)
    file_name = os.path.join(kepid_dir,
                             "{0:09d}_plnt_num-{1:02d}_tce.record".format(int(tce_data.kepid), tce_data.tce_plnt_num))

    try:
        with open(file_name, 'rb') as fp:
            result_X = pickle.load(fp)
        fp.close()
    except IOError:
        print("Could not read file: ", file_name)

    # print(X1)
    # print(X2)
    return result_X


# This function will get the '.record' file directly to read the data
# It is to be used in the training generator function, as the files
# will be stored in the training folder
def read_tce_global_view_local_view_from_file(record_file):
    # result_X = []
    X1, X2 = [], []

    try:
        with open(record_file, 'rb') as fp:
            # result_X = pickle.load(fp)
            X1, X2 = pickle.load(fp)
        fp.close()
    except IOError:
        print("Could not read file: ", record_file)

    # print(X1)
    # print(X2)
    # return result_X
    return X1, X2
