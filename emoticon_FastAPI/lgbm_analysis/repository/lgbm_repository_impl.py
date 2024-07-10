import os
import random
from collections import Counter

import joblib
import optuna
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier

from emoticon_FastAPI.lgbm_analysis.repository.lgbm_repository import LgbmAnalysisRepository


class LgbmAnalysisRepositoryImpl(LgbmAnalysisRepository):
    CLASSES = ['귀여운', '재밌는', '메시지']
    CSV_SAVE_PATH = 'filtered_data_{age_group}_{gender}.csv'

    def readData(self):
        load_dotenv()
        MYSQL_HOST = os.getenv('MYSQL_HOST')
        MYSQL_PORT = os.getenv('MYSQL_PORT')
        MYSQL_USER = os.getenv('MYSQL_USER')
        MYSQL_PASSWORD = quote_plus(os.getenv('MYSQL_PASSWORD'))
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")

        query_report = "SELECT age, gender, account_id FROM report"
        query_orders = "SELECT id as 'orders_id', account_id FROM orders"
        query_orders_item = "SELECT orders_id, product_id FROM orders_item"
        query_product = "SELECT productId as 'product_id', productCategory as 'target' FROM product"

        df_report = pd.read_sql(query_report, engine)
        df_orders = pd.read_sql(query_orders, engine)
        df_orders_item = pd.read_sql(query_orders_item, engine)
        df_product = pd.read_sql(query_product, engine)

        df_orders_report = pd.merge(df_orders, df_report, on='account_id')
        df_orders_report_orders_item = pd.merge(df_orders_report, df_orders_item, on='orders_id')
        df_full = pd.merge(df_orders_report_orders_item, df_product, on='product_id')

        df_final = df_full[['age', 'gender', 'target']].copy()
        return df_final

    def featureEncoding(self, dataFrame, forTrain):
        dataFrame['age_group'] = pd.cut(dataFrame['age'], bins=[0, 19, 29, 39, 49, 59, float('inf')],
                                  labels=[10, 20, 30, 40, 50, 99], right=False)
        dataFrame['age_group'] = dataFrame['age_group'].astype(int)
        dataFrame['gender'] = dataFrame['gender'].apply(lambda x: 1 if x == '남성' else 0)

        if forTrain:
            dataFrame['target'] = dataFrame['target'].apply(lambda x: 0 if x == '귀여운' else (1 if x == '재밌는' else 2))

        return dataFrame

    def saveSplitDataFrame(self, encoded_df):
        age_groups = encoded_df['age_group'].value_counts().index.tolist()
        genders = encoded_df['gender'].unique().tolist()

        for age_group in age_groups:
            for gender in genders:
                filtered_df = encoded_df[(encoded_df['age_group'] == age_group) & (encoded_df['gender'] == gender)]

                if not filtered_df.empty:
                    # CSV 파일로 저장
                    csv_path = self.CSV_SAVE_PATH.format(age_group=age_group, gender=gender)
                    filtered_df.to_csv(csv_path, index=False)
                    print(f'Filtered data saved to {csv_path}. ')
    def readFilterData(self, age_group, gender):
        path = self.CSV_SAVE_PATH.format(age_group=age_group, gender=gender)
        print('path', path)
        df = pd.read_csv(path)
        print(f' {age_group} {gender} csv load : \n', df)
        return df

    def splitFeatureTarget(self, dataFrame):
        X = dataFrame[['age', 'gender', 'age_group']]
        y = dataFrame['target']
        return X, y

    def featureScale(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def trainTestSplit(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def smote(self, X_train, y_train):
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())

        n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

        smote = SMOTE(k_neighbors=n_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

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

    async def getScore(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy: {:.4f}'.format(acc))

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print('Confusion Matrix:\n', cm)
        print('Classification Report:\n', cr)

    async def predictModel(self, trainedModel, X):
        predictProbability = trainedModel.predict_proba(X)
        print('각각 카테고리별 확률 : ', predictProbability)

        topClass = trainedModel.predict(X)[0]
        print('predictions : ', topClass)

        return self.CLASSES[topClass-1], predictProbability

    def getProductOfCategory(self, category, k):
        print('repository -> getProductOfCategory()')
        load_dotenv()
        MYSQL_HOST = os.getenv('MYSQL_HOST')
        MYSQL_PORT = os.getenv('MYSQL_PORT')
        MYSQL_USER = os.getenv('MYSQL_USER')
        MYSQL_PASSWORD = quote_plus(os.getenv('MYSQL_PASSWORD'))
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        query_report = "select age,gender, account_id from report"
        query_orders = "select id as orders_id,account_id from orders"
        query_product = "SELECT productId as 'product_id', productCategory as 'target' FROM product"
        query_orders_item = "SELECT orders_id, product_id FROM orders_item"
        df_product = pd.read_sql(query_product, engine)
        df_orders_item = pd.read_sql(query_orders_item, engine) # orders_id를 기준으로 prdocut_id count
        df_report = pd.read_sql(query_report, engine)
        df_orders = pd.read_sql(query_orders, engine)  # orders_id를 기준으로 prdocut_id count


        orders_df = df_orders_item.groupby(['product_id'])[['orders_id']].count().reset_index()
        orders_df = orders_df.rename(columns={'orders_id': 'cnt'})
        orders_df = orders_df.sort_values(by='cnt', ascending=False).reset_index(drop=True)
        # print(orders_df)

        category_product = df_product[df_product['target'] == category]
        d1 = category_product.merge(df_orders_item,on='product_id',how='inner')
        # print(1111,d1)
        product_names = category_product.merge(orders_df, on='product_id', how='inner')
        # print(product_names)
        product_names = product_names[['product_id', 'cnt']]
        product_names = product_names.sort_values(by='cnt', ascending=False)['product_id'].tolist()


        selected_product_ids = product_names[:min(k, len(product_names))]
        print('selected_product_ids: ', selected_product_ids)

        return selected_product_ids

    def getProductData(self):
        load_dotenv()
        MYSQL_HOST = os.getenv('MYSQL_HOST')
        MYSQL_PORT = os.getenv('MYSQL_PORT')
        MYSQL_USER = os.getenv('MYSQL_USER')
        MYSQL_PASSWORD = quote_plus(os.getenv('MYSQL_PASSWORD'))
        MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')

        engine = create_engine(
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
        query_report = "select age,gender, account_id from report"
        query_orders = "select id as 'orders_id',account_id from orders"
        query_product = "SELECT productId as 'product_id', productCategory as 'target' FROM product"
        query_orders_item = "SELECT orders_id, product_id FROM orders_item"
        df_product = pd.read_sql(query_product, engine)
        df_orders_item = pd.read_sql(query_orders_item, engine)  # orders_id를 기준으로 prdocut_id count
        df_report = pd.read_sql(query_report, engine)
        df_orders = pd.read_sql(query_orders, engine)

        profile = df_report.merge(df_orders, on='account_id', how='inner')
        df_orders_item = df_orders_item.merge(profile, on="orders_id", how='inner')

        # category_product = df_product[df_product['target'] == category]
        d1 = df_product.merge(df_orders_item, on='product_id', how='inner')
        d2 = d1[['age', 'target', 'gender']]
        return d2