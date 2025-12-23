from src.MLOps_Project.config.configuration import Configuration_Manager
from src.MLOps_Project.components.model_evaluation import Model_Evaluation
from src.MLOps_Project import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
            config = Configuration_Manager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = Model_Evaluation(config=model_evaluation_config)
            model_evaluation.log_into_mlflow()      
            
if __name__ == "__main__":
        try:
            logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
            model_evaluation_pipeline = ModelEvaluationPipeline()
            model_evaluation_pipeline.initiate_model_evaluation()
            logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n")
        except Exception as e:
            logger.exception(f"Error in {STAGE_NAME}: {e}")
            raise e