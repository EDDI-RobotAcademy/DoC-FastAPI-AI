import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

from emoticon_FastAPI.lgbm_analysis.repository.lgbm_repository_impl import LgbmAnalysisRepositoryImpl
from emoticon_FastAPI.lgbm_analysis.service.lgbm_service import LgbmAnalysisService


class LgbmAnalysisServiceImpl(LgbmAnalysisService):
    SAVED_MODEL_PATH = 'lgbm_classifier_model.pkl'

    def __init__(self):
        self.__lgbmAnalysisRepository = LgbmAnalysisRepositoryImpl()

    def saveTrainedModel(self, trainedModel, modelPath):
        joblib.dump(trainedModel, modelPath)

    async def lgbmTrain(self):
        print('service -> lgbmTrain()')
        df = self.__lgbmAnalysisRepository.readData() # await
        encoded_df = self.__lgbmAnalysisRepository.featureEncoding(df, forTrain=True)
        X, y = self.__lgbmAnalysisRepository.splitFeatureTarget(encoded_df)
        num_classes = len(np.unique(y))

        X_scaled = self.__lgbmAnalysisRepository.featureScale(X)
        X_train, X_test, y_train, y_test = self.__lgbmAnalysisRepository.trainTestSplit(X_scaled, y)
        selectedModel = await self.__lgbmAnalysisRepository.selectLGBMmodel(num_classes)
        trainedModel = await self.__lgbmAnalysisRepository.trainModel(selectedModel, X_train, y_train)
        self.saveTrainedModel(trainedModel, self.SAVED_MODEL_PATH)
        await self.__lgbmAnalysisRepository.getAccuracy(trainedModel,  X_test, y_test)

        return True
