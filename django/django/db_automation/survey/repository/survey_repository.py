from abc import ABC, abstractmethod



class SurveyRepository(ABC):

    @abstractmethod
    def getMaxId(self):
        pass


    @abstractmethod
    def registerSurvey(self, surveyId):
        pass


    @abstractmethod
    def findSurvey(self, surveyId):
        pass


    @abstractmethod
    def getAllRandomString(self):
        pass


    @abstractmethod
    def findSurveyIdByRandomString(self, randomString):
        pass


    @abstractmethod
    def findRandomStringBySurveyId(self, surveyId):
        pass