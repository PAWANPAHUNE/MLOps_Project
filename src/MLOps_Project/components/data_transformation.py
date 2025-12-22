import os
from src.MLOps_Project import logger
from sklearn.model_selection import train_test_split
import pandas as pd 
from src.MLOps_Project.config.configuration import Configuration_Manager
from src.MLOps_Project.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
            data = pd.read_csv(self.config.data_path)
            train, test = train_test_split(data, test_size=0.2, random_state=42)

            train_data_path = os.path.join(self.config.root_dir, "train.csv")
            test_data_path = os.path.join(self.config.root_dir, "test.csv")

            train.to_csv(train_data_path, index=False)
            test.to_csv(test_data_path, index=False)

            logger.info(f"Data transformation completed. Train and test data saved at {self.config.root_dir}")
            logger.info(train.shape)
            logger.info(test.shape)