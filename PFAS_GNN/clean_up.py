from libs.utils import setup_logging, remove_predictions, cleanup_models, remove_torch_geometry_garbage, cleanup_temp_files

if __name__ == '__main__':
    logger = setup_logging()
   # remove_predictions()
   # cleanup_models()
   # remove_torch_geometry_garbage()
    cleanup_temp_files()
    logger.info("All predictions, models and torch geometry garbage have been removed.")