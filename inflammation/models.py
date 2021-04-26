"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :returns: A numpy array containing the data
    from the file provided by the filename
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array.

    Arithmetic mean along the zeroth axis

    :param data: A 2d numpy array
    :returns: A 1d numpy array of means
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array.

    Selects the maximum value along the zeroth axis

    :param: A 2d numpy array
    :returns: A 1d numpy array of maximums
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array.

    Selects the minimum value along the zeroth axis

    :param: A 2d numpy array
    :returns: A 1d numpy array of minimums
    """
    return np.min(data, axis=0)


# TODO(lesson-design) Add Patient class
# TODO(lesson-design) Implement data persistence
# TODO(lesson-design) Add Doctor class
