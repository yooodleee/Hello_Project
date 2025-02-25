from abc import ABC, abstractmethod



class InterviewResultService(ABC):

    @abstractmethod
    def saveInterviewResult(self, scoreReusltList, accountId):
        pass

    @abstractmethod
    def getInterviewResult(self, accountId):
        pass