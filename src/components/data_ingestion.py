import os
import sys
from src.exception import CustomException
from src.logger import log
from src import utils
import pandas as pd
from dataclasses import dataclass
from src.components import model_trainer
from src.pipeline import predict_pipeline


@dataclass
class DataIngestionConfig:
    project_path = utils.getPath()
    raw_data_path: str = os.path.join(project_path, 'artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        log('Entered the data ingestion method or component')
        try:
            que, ans = [], []
            with open(os.path.join(self.ingestion_config.project_path, 'data', 'dialogs.txt'), 'r') as f:
                for line in f:
                    line = line.split('\t')
                    que.append(line[0])
                    ans.append(line[1].replace('\n', ''))

            df = pd.DataFrame({'que':que,'ans':ans})
            log('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            log('Ingestion of the data is completed')

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    raw_data_path = DataIngestion().initiate_data_ingestion()
    log('raw_data_path' +raw_data_path)
    model_trainer.ModelTrainer().initiate_model_trainer()

    # predict_pipeline.PredictPipeline().load_model()
    # resp = predict_pipeline.PredictPipeline().ask("dfghjk")
    # print(resp)