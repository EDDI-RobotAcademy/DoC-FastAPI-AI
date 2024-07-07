import numpy as np
import mysql.connector
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data_analysis.repository.analysis_repository import AnalysisRepository

config = {
    'user': 'eddi',
    'password': 'eddi@123',
    'host': 'localhost',
    'database': 'emoticon_db',
}

class AnalysisRepositoryImpl(AnalysisRepository):

    def __init__(self):
        self.label_encoder_age_group = LabelEncoder()
        self.label_encoder_gender = LabelEncoder()
        self.label_encoder_productId = LabelEncoder()

    async def readData(self):
        conn = mysql.connector.connect(**config)

        query_report = "SELECT * FROM report"
        query_orders = "SELECT * FROM orders"
        query_orders_item = "SELECT * FROM orders_item"

        report = pd.read_sql(query_report, conn)
        orders = pd.read_sql(query_orders, conn)
        orders_item = pd.read_sql(query_orders_item, conn)

        conn.close()

        orders_report = pd.merge(orders, report, on='account_id')

        # orders_item과 orders_report를 orders_id를 기준으로 병합
        full = pd.merge(orders_report, orders_item, left_on='id_x', right_on='orders_id')

        # 필요한 열 선택 (account_id, 나이대, 성별, 상품)
        df_final = full[['account_id', 'age', 'gender', 'product_id']].copy()
        return df_final

    def transfromData(self, data):
        bins = [0, 20, 30, 40, 50, 60, 100]
        labels = ['10대', '20대', '30대', '40대', '50대', '60대 이상']  # 레이블 설정
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
        data = data[['account_id', 'age_group', 'gender', 'product_id']]

        return data

    def encoding(self, data, col):
        for column in col:
            if column == 'age_group':
                data[column] = self.label_encoder_age_group.fit_transform(data[column])
            elif column == 'gender':
                data[column] = self.label_encoder_gender.fit_transform(data[column])
            elif column == 'product_id':
                data[column] = self.label_encoder_productId.fit_transform(data[column])
        X = ['age_group', 'gender']
        y = 'product_id'
        return data[X], data[y]

    def dataSetting(self, x, y):
        return lgb.Dataset(x, label=y)

    def trainModel(self, y, data):
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),  # 예측할 클래스의 수
            'metric': 'multi_logloss',
            'is_unbalance': True,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9
        }

        model = lgb.train(params, data, num_boost_round=100)
        return model


    def predict(self, age_group, gender, model):
        try:
            input_data = pd.DataFrame({'age_group': [age_group], 'gender': [gender]})
            input_data['age_group'] = self.label_encoder_age_group.fit_transform(input_data['age_group'])
            input_data['gender'] = self.label_encoder_gender.fit_transform(input_data['gender'])
            if model != None:
                middle_result = model.predict(input_data)
                result = np.argsort(middle_result[0])[::-1][:5]
                product_names = [self.label_encoder_productId.inverse_transform([cls])[0] for cls in result]
                return product_names
                print(product_names)
        except Exception as e:
            print(f'오류발생: {e}')