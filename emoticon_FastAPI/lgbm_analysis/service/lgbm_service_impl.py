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
        df = self.__lgbmAnalysisRepository.readData()
        # print(df.groupby(['target'])[['age']].mean())
        # print(df.groupby(['target'])[['target']].count())
        # print('age: ', df['age'].describe())
        encoded_df = self.__lgbmAnalysisRepository.featureEncoding(df, forTrain=True)
        # print('df : ', df)
        # print(df.head())
        # print(df.describe())
        print('encoded_df', encoded_df.head())
        print('encoded_df describe: ',encoded_df.describe())
        # print(df['age'].value_counts())
        # print(df['target'].value_counts())
        # print(df['gender'].value_counts())
        X, y = self.__lgbmAnalysisRepository.splitFeatureTarget(encoded_df)
        num_classes = len(np.unique(y))

        X_scaled = self.__lgbmAnalysisRepository.featureScale(X)
        print('scaled X: ', X_scaled)
        X_train, X_test, y_train, y_test = self.__lgbmAnalysisRepository.trainTestSplit(X_scaled, y)
        X_train_smote, y_train_smote = self.__lgbmAnalysisRepository.smote(X_train, y_train)
        selectedModel = await self.__lgbmAnalysisRepository.selectLGBMmodel(num_classes)
        trainedModel = await self.__lgbmAnalysisRepository.trainModel(selectedModel, X_train_smote, y_train_smote)
        self.saveTrainedModel(trainedModel, self.SAVED_MODEL_PATH)
        await self.__lgbmAnalysisRepository.getScore(trainedModel,  X_test, y_test)

        return True

    async def lgbmPredict(self, age, gender):
        loaded_model = self.__lgbmAnalysisRepository.loadModel(self.SAVED_MODEL_PATH)
        print('로드된 모델 : ', loaded_model)
        X_new = pd.DataFrame({'age': [age], 'gender': [gender]})
        encoded_X = self.__lgbmAnalysisRepository.featureEncoding(X_new, forTrain=False)
        print('encoding된 X: ', encoded_X)
        # scaled_X = self.__lgbmAnalysisRepository.featureScale(encoded_X)
        # print('scaled X: ', scaled_X)
        predicted_class, probability = await self.__lgbmAnalysisRepository.predictModel(loaded_model, encoded_X, 3)
        print('prediction: ', predicted_class)
        return predicted_class, probability


    def getRecommendProducts(self, category, k):
        return self.__lgbmAnalysisRepository.getProductOfCategory(category, k=5)
