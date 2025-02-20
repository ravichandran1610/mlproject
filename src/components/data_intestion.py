import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.intestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered into data ingestion method or compoenet')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.intestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.intestion_config.raw_data_path, index=False, header=True)

            logging.info('Train and Test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)

            train_set.to_csv(self.intestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.intestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return(
                self.intestion_config.train_data_path,
                self.intestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
