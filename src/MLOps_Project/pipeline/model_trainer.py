from src.MLOps_Project.config.configuration import Configuration_Manager
from src.MLOps_Project.components.model_trainer import Model_Trainer
from src.MLOps_Project import logger

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    
    def __init__(self):
        pass

    def initiate_model_trainer(self):
            config = Configuration_Manager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = Model_Trainer(config=model_trainer_config)
            model_trainer.train_model()

if __name__ == "__main__":
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            model_trainer_pipeline = ModelTrainerPipeline()
            model_trainer_pipeline.initiate_model_trainer()
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e