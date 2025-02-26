from abc import ABC, abstractmethod



class SurveyQuestionRepository(ABC):

    @abstractmethod
    def registerQuestion(self, survey, questionTitle, questionType, essential, images):
        pass


    @abstractmethod
    def findQuestion(self, questionId):
        pass


    @abstractmethod
    def getQuestionBySurveyId(self, surveyId):
        pass