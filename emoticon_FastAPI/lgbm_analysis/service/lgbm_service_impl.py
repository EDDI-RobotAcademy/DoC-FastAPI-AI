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

    async def lgbmPredict(self, age, gender):
        loaded_model = self.__lgbmAnalysisRepository.loadModel(self.SAVED_MODEL_PATH)
        X_new = pd.DataFrame({'age': [age], 'gender': [gender]})
        encoded_X = self.__lgbmAnalysisRepository.featureEncoding(X_new, forTrain=False)
        scaled_X = self.__lgbmAnalysisRepository.featureScale(encoded_X)
        predicted_class = await self.__lgbmAnalysisRepository.predictModel(loaded_model, scaled_X, 3)
        print('prediction: ', predicted_class)
        return predicted_class


    def getRecommendProducts(self, category, k):
        return self.__lgbmAnalysisRepository.getProductOfCategory(category, k=5)
