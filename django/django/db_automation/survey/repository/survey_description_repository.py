from abc import ABC, abstractmethod



class SurveyDescriptionRepository(ABC):

    @abstractmethod
    def registerDescription(self, survey, surveyDescription):
        pass


    @abstractmethod
    def getDescriptionBySurveyId(self, surveyId):
        pass