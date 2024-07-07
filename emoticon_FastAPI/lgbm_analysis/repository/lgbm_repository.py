from abc import ABC, abstractmethod

class LgbmAnalysisRepository(ABC):

    @abstractmethod
    def readData(self):
        pass

    @abstractmethod
    def featureEncoding(self, dataFrame, forTrain):
        pass

    @abstractmethod
    def splitFeatureTarget(self, dataFrame):
        pass

    @abstractmethod
    def featureScale(self, X):
        pass # scaled_X

    @abstractmethod
    def trainTestSplit(self, X, y):
        pass

    @abstractmethod
    def selectLGBMmodel(self, num_classes):
        pass
    @abstractmethod
    def trainModel(self, model, X_train, y_train):
        pass

    @abstractmethod
    def loadModel(self, modelPath):
        pass

    @abstractmethod
    def getAccuracy(self, model, X_test, y_test):
        pass

    @abstractmethod
    def predictModel(self, trainedModel, X, top_k):
        pass
    @abstractmethod
    def getProductOfCategory(self, category, k):
        pass



