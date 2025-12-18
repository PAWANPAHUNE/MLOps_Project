from src.MLOps_Project import logger
from src.MLOps_Project.pipeline.data_ingestion import DataIngestionTrainingPipeline 

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
    
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise e