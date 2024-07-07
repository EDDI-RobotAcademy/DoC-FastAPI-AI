from abc import ABC, abstractmethod


class LgbmAnalysisService(ABC):
    @abstractmethod
    def saveTrainedModel(self, trainedModel, modelPath):
        pass

    @abstractmethod
    def lgbmTrain(self):
        pass
