from abc import ABC, abstractmethod



class SurveySelectionRepository(ABC):

    @abstractmethod
    def registerSelection(self, question, selection):
        pass


    @abstractmethod
    def getSelectionByQuestionId(self, questionId):
        pass


    @abstractmethod
    def findSelectionBySelectionId(self, selection):
        pass


    @abstractmethod
    def findSelectionBySelectionName(self, question, selectionName):
        pass