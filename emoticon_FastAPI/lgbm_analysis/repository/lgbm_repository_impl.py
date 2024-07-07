import os
import random

import joblib
import optuna
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

from emoticon_FastAPI.lgbm_analysis.repository.lgbm_repository import LgbmAnalysisRepository


class LgbmAnalysisRepositoryImpl(LgbmAnalysisRepository):
    CLASSES = ['귀여운', '재밌는', '메시지']

    def readData(self):
        load_dotenv()
        # 환경 변수 가져오기
        MYSQL_HOST = os.getenv('MYSQL_HOST')
        MYSQL_PORT = os.getenv('MYSQL_PORT')
        MYSQL_USER = os.getenv('MYSQL_USER')
        MYSQL_PASSWORD = quote_plus(os.getenv('MYSQL_PASSWORD'))
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

        # SQLAlchemy 엔진 생성
        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")

        # 각 테이블을 DataFrame으로 읽기
        query_report = "SELECT age, gender, account_id FROM report"
        query_orders = "SELECT id as 'orders_id', account_id FROM orders"
        query_orders_item = "SELECT orders_id, product_id FROM orders_item"
        query_product = "SELECT productId as 'product_id', productCategory as 'target' FROM product"

        df_report = pd.read_sql(query_report, engine)
        df_orders = pd.read_sql(query_orders, engine)
        df_orders_item = pd.read_sql(query_orders_item, engine)
        df_product = pd.read_sql(query_product, engine)

        # merge
        df_orders_report = pd.merge(df_orders, df_report, on='account_id')
        df_orders_report_orders_item = pd.merge(df_orders_report, df_orders_item, on='orders_id')
        df_full = pd.merge(df_orders_report_orders_item, df_product, on='product_id')

        df_final = df_full[['age', 'gender', 'target']].copy()

        return df_final


    def featureEncoding(self, dataFrame, forTrain):
        # # 나이를 연령대로 변환
        dataFrame['age'] = pd.cut(dataFrame['age'], bins=[0, 19, 29, 39, 49, 59, float('inf')],
                                 labels=[10, 20, 30, 40, 50, 99], right=False)
        # 성별을 0 1로 변환
        dataFrame['gender'] = dataFrame['gender'].apply(lambda x: 1 if x == '남성' else 0)

        if forTrain ==True:
            # 귀여운 = 0, 재밌는 : 1, 메시지: 2
            dataFrame['target'] = dataFrame['target'].apply(lambda x: 0 if x == '귀여운' else (1 if x == '재밌는' else 2))

        return dataFrame

    def splitFeatureTarget(self, dataFrame):
        X = dataFrame[['age', 'gender']]
        y = dataFrame['target']

        return X, y

    def featureScale(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled


    def trainTestSplit(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)


    async def selectLGBMmodel(self, num_classes):
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,  # 예측할 클래스의 수
            'metric': 'multi_logloss',
            'is_unbalance': True,
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.04948730309161483,
            'feature_fraction': 0.9,
            'n_estimators': 100,
            'force_col_wise': True
        }
        model = LGBMClassifier(**params)
        return model

    async def trainModel(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def loadModel(self, modelPath):
        return joblib.load(modelPath)

    async def getAccuracy(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('accuracy: {:.4f}'.format(acc))
