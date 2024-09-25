import numpy as np


def nse(observed, simulated):
    return 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)
def pbias(observed, simulated):
    return ((np.sum(simulated - observed) / np.sum(observed)) * 100)
def mse(observed, simulated):
    return np.mean((observed - simulated)**2)
def rmse(observed, simulated):
    return np.sqrt(mse(observed, simulated))

def mape(observed, simulated):
    nonzero_observed = observed[observed != 0]
    nonzero_simulated = simulated[observed != 0]
    return np.mean(np.abs((nonzero_observed - nonzero_simulated) / nonzero_observed)) * 100

