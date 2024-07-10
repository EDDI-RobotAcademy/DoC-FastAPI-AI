from abc import ABC, abstractmethod


class LgbmAnalysisService(ABC):
    @abstractmethod
    def saveTrainedModel(self, trainedModel, modelPath):
        pass

    @abstractmethod
    def lgbmTrain(self):
        pass

    @abstractmethod
    def lgbmPredict(self, age, gender):
        pass
    @abstractmethod
    def getRecommendProducts(self, category, k):
        pass

    @abstractmethod
    def getVisualData(self):
        pass