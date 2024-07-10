import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from emoticon_FastAPI.lgbm_analysis.repository.lgbm_repository_impl import LgbmAnalysisRepositoryImpl
from emoticon_FastAPI.lgbm_analysis.service.lgbm_service import LgbmAnalysisService

class LgbmAnalysisServiceImpl(LgbmAnalysisService):
    SAVED_MODEL_PATH = 'lgbm_classifier_model_{age_group}_{gender}.pkl'  # 모델 저장 경로

    def __init__(self):
        self.__lgbmAnalysisRepository = LgbmAnalysisRepositoryImpl()

    def saveTrainedModel(self, trainedModel, modelPath):
        joblib.dump(trainedModel, modelPath)

    async def lgbmTrain(self):
        print('service -> lgbmTrain()')
        df = self.__lgbmAnalysisRepository.readData()
        encoded_df = self.__lgbmAnalysisRepository.featureEncoding(df, forTrain=True)
        age_groups = encoded_df['age_group'].value_counts().index.tolist()
        genders = encoded_df['gender'].value_counts().index.tolist()
        print('age_groups', age_groups)
        print('genders', genders)
        self.__lgbmAnalysisRepository.saveSplitDataFrame(encoded_df)
        for age_group in age_groups:
            for gender in genders:
                filtered_df = self.__lgbmAnalysisRepository.readFilterData(age_group, gender)
                X, y = self.__lgbmAnalysisRepository.splitFeatureTarget(filtered_df)
                num_classes = len(np.unique(y))
                X_scaled = self.__lgbmAnalysisRepository.featureScale(X)
                X_train, X_test, y_train, y_test = self.__lgbmAnalysisRepository.trainTestSplit(X_scaled, y)

                if filtered_df['target'].nunique() == 3 and filtered_df['target'].value_counts().min() > 1:
                    X_train_smote, y_train_smote =  self.__lgbmAnalysisRepository.smote(X_train, y_train)
                else:
                    X_train_smote, y_train_smote = X_train, y_train

                selectedModel = await self.__lgbmAnalysisRepository.selectLGBMmodel(num_classes)
                trainedModel = await self.__lgbmAnalysisRepository.trainModel(selectedModel, X_train_smote, y_train_smote)
                model_path = self.SAVED_MODEL_PATH.format(age_group=age_group, gender=gender)
                self.saveTrainedModel(trainedModel, model_path)
                await self.__lgbmAnalysisRepository.getScore(trainedModel, X_test, y_test)

    async def lgbmPredict(self, age, gender):
        X_new = pd.DataFrame({'age': [age], 'gender': [gender]})
        encoded_X = self.__lgbmAnalysisRepository.featureEncoding(X_new, forTrain=False)
        print('encoded_x: ', encoded_X)
        age_group = encoded_X['age_group'][0]
        gender = encoded_X['gender'][0]
        model_path = self.SAVED_MODEL_PATH.format(age_group=age_group, gender=gender)
        loaded_model = self.__lgbmAnalysisRepository.loadModel(model_path)
        print('model_path', model_path)
        predicted_class, probability = await self.__lgbmAnalysisRepository.predictModel(loaded_model, encoded_X)
        print(f'prediction: {predicted_class}, probability: {probability}')
        return predicted_class, probability

    def getRecommendProducts(self, category, k):
        return self.__lgbmAnalysisRepository.getProductOfCategory(category, k)

    def getVisualData(self):
        return self.__lgbmAnalysisRepository.getProductData()