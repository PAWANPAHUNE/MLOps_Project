from src.MLOps_Project.config.configuration import Configuration_Manager
from src.MLOps_Project.components.data_transformation import DataTransformation
from src.MLOps_Project import logger

STAGE_NAME = "Data Transformation Stage"
class DataTransformationTrainingPipeline:
    
    def __init__(self):
        pass
    def initiate_data_transformation(self):
            config = Configuration_Manager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.train_test_splitting()
if __name__ == "__main__":
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            data_transformation_pipeline = DataTransformationTrainingPipeline()
            data_transformation_pipeline.initiate_data_transformation()
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e
        