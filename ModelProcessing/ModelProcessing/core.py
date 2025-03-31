from ModelProcessing.processing_program import ProcessingProgram
from ModelProcessing.find_VPUID import find_VPUID
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def process_SCV_SWATGenXModel(config):
    if config['VPUID'] is None:
        config['VPUID'] = find_VPUID(config['NAME'])
        logging.info(f"VPUID: {config['VPUID']}")
        logging.info(f"NAME: {config['NAME']}")
        logging.info(f"LEVEL: {config['LEVEL']}")
        logging.info(f"MODEL_NAME: {config['MODEL_NAME']}")
        logging.info(f"START_YEAR: {config['START_YEAR']}")
        logging.info(f"END_YEAR: {config['END_YEAR']}")
        logging.info(f"nyskip: {config['nyskip']}")
        logging.info(f"sensitivity_flag: {config['sensitivity_flag']}")
        logging.info(f"calibration_flag: {config['calibration_flag']}")
        logging.info(f"verification_flag: {config['verification_flag']}")
        logging.info(f"sen_total_evaluations: {config['sen_total_evaluations']}")
        logging.info(f"sen_pool_size: {config['sen_pool_size']}")
        logging.info(f"num_levels: {config['num_levels']}")
        logging.info(f"cal_pool_size: {config['cal_pool_size']}")
        logging.info(f"max_cal_iterations: {config['max_cal_iterations']}")
        logging.info(f"termination_tolerance: {config['termination_tolerance']}")
        logging.info(f"epsilon: {config['epsilon']}")
        logging.info(f"Ver_START_YEAR: {config['Ver_START_YEAR']}")
        logging.info(f"Ver_END_YEAR: {config['Ver_END_YEAR']}")
        logging.info(f"Ver_nyskip: {config['Ver_nyskip']}")
    
    SCV_args = ProcessingProgram(
			config
    )

    SCV_args.SWATGenX_SCV()
