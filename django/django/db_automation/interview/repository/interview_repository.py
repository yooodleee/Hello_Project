from abc import ABC, abstractmethod




class InterviewRepository(ABC):
    
    @abstractmethod
    def getMaxId(self):
        pass


    @abstractmethod
    def insertData(self, interviewId, questionList):
        pass


    @abstractmethod
    def insertFirstQuestion(self, question):
        pass


    @abstractmethod
    def insertTechQuestion(self, question, job):
        pass


    @abstractmethod
    def getData(self, sessionId):
        pass


    @abstractmethod
    def getFirstQuestion(self, questionId):
        pass


    @abstractmethod
    def getTechQuestion(self, job):
        pass