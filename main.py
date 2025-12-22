from src.MLOps_Project import logger
from src.MLOps_Project.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.MLOps_Project.pipeline.data_validation import DataValidationTrainingPipeline
from src.MLOps_Project.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.MLOps_Project.pipeline.model_trainer import ModelTrainerPipeline

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
    
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.initiate_data_validation()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    model_trainer_pipeline = ModelTrainerPipeline()
    model_trainer_pipeline.initiate_model_trainer()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(f"Error in {STAGE_NAME}: {e}")
    raise e

