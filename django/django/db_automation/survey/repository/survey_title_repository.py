from abc import ABC, abstractmethod



class SurveyTitleRepository(ABC):

    @abstractmethod
    def registerTitle(self, survey, surveyTitle):
        pass


    @abstractmethod
    def getAllTitles(self):
        pass


    @abstractmethod
    def getTitleBySurveyId(self, surveyId):
        pass