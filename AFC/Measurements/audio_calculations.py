import numpy as np

def calc_rms(signal_arr):
    return np.sqrt(np.mean(signal_arr**2))

def calc_db(scalar):
    return 20*np.log10(scalar)

def calc_dbrms(signal_arr):
    return calc_db(calc_rms(signal_arr))

def calc_scalar(db):
    return 10 ** (db/20)