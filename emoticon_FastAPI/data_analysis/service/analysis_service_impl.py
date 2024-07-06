import pandas as pd
import numpy as np

from data_analysis.repository.analysis_repository_impl import AnalysisRepositoryImpl
from data_analysis.service.analysis_service import AnalysisService


class AnalysisServiceImpl(AnalysisService):
    dataCol = ['age_group', 'gender', 'product_id']

    def __init__(self):
        self.__AnalysisRepository = AnalysisRepositoryImpl()
        self.model = None

    async def trainModel(self):
        rawData = await self.__AnalysisRepository.readData()
        transData = self.__AnalysisRepository.transfromData(rawData)
        x,y = self.__AnalysisRepository.encoding(transData,self.dataCol)
        trainData = self.__AnalysisRepository.dataSetting(x,y)
        model = self.__AnalysisRepository.trainModel(y,trainData)

    async def predict(self, model):
        age = "20대"
        gender = "여성"
        predict = self.__AnalysisRepository.predict(age, gender, model)
        return predict