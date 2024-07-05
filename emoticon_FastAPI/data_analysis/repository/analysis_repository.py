from abc import ABC, abstractmethod


class AnalysisRepository(ABC):

    @abstractmethod
    def readData(self):
        pass

    @abstractmethod
    def transfromData(self,data):
        pass

    @abstractmethod
    def encoding(self,data,col):
        pass

    @abstractmethod
    def dataSetting(self,x,y):
        pass

    @abstractmethod
    def trainModel(self,y,data):
        pass

    @abstractmethod
    def predict(self,age,gender, model):
        pass