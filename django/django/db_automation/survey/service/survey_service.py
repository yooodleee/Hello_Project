from abc import ABC, abstractmethod



class SurveyService(ABC):

    @abstractmethod
    def createSurveyForm(self, randomString):
        pass


    @abstractmethod
    def getSurveyBySurveyId(self, surveyId):
        pass


    @abstractmethod
    def getQuestionByQuestionId(self, questionId):
        pass


    @abstractmethod
    def registerTitleDescription(self, survey, surveyTitle, surveyDescription):
        pass


    @abstractmethod
    def registerQuestion(self, survey, questionTitle, questionType, essential, images):
        pass


    @abstractmethod
    def registerSelection(self, question, selection):
        pass


    @abstractmethod
    def getSurveyList(self):
        pass


    @abstractmethod
    def getRandomStringList(self):
        pass


    @abstractmethod
    def getServeyById(self, surveyId):
        pass


    @abstractmethod
    def saveAnswer(self, answers, accountId):
        pass


    @abstractmethod
    def getSurveyIdByRandomString(self, randomString):
        pass


    @abstractmethod
    def getResultById(self, surveyId):
        pass


    @abstractmethod
    def getAnswerByAccountId(self, accountId):
        pass