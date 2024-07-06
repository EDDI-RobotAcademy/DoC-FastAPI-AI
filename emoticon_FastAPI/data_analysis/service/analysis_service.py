from abc import ABC, abstractmethod


class AnalysisService(ABC):

    @abstractmethod
    def trainModel(self):
        pass

    @abstractmethod
    def predict(self, model):
        pass