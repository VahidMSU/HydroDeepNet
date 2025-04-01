import numpy as np
from ModelProcessing.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def nse(observed, simulated):
    """Calculate Nash-Sutcliffe Efficiency coefficient.
    
    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        
    Returns:
        float: NSE coefficient
    """
    try:
        result = 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)
        logger.debug(f"NSE calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating NSE: {str(e)}")
        return None

def pbias(observed, simulated):
    """Calculate percent bias.
    
    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        
    Returns:
        float: Percent bias
    """
    try:
        result = ((np.sum(simulated - observed) / np.sum(observed)) * 100)
        logger.debug(f"PBIAS calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating PBIAS: {str(e)}")
        return None

def mse(observed, simulated):
    """Calculate mean squared error.
    
    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        
    Returns:
        float: Mean squared error
    """
    try:
        result = np.mean((observed - simulated)**2)
        logger.debug(f"MSE calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating MSE: {str(e)}")
        return None

def rmse(observed, simulated):
    """Calculate root mean squared error.
    
    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        
    Returns:
        float: Root mean squared error
    """
    try:
        result = np.sqrt(mse(observed, simulated))
        logger.debug(f"RMSE calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating RMSE: {str(e)}")
        return None

def mape(observed, simulated):
    """Calculate mean absolute percentage error.
    
    Args:
        observed: Array of observed values
        simulated: Array of simulated values
        
    Returns:
        float: Mean absolute percentage error
    """
    try:
        nonzero_observed = observed[observed != 0]
        nonzero_simulated = simulated[observed != 0]
        
        if len(nonzero_observed) == 0:
            logger.warning("No non-zero observed values for MAPE calculation")
            return None
            
        result = np.mean(np.abs((nonzero_observed - nonzero_simulated) / nonzero_observed)) * 100
        logger.debug(f"MAPE calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating MAPE: {str(e)}")
        return None

