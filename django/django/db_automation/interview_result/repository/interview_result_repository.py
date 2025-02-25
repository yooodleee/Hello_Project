from abc import ABC, abstractmethod



class InterviewResultRepository(ABC):

    @abstractmethod
    def registerInterviewResult(self, account):
        pass


    @abstractmethod
    def registerInterviewResultQAS(self, interviewResult, scoreResultList):
        pass

    
    @abstractmethod
    def getLastInterviewResult(self):
        pass


    @abstractmethod
    def getLastInterviewResultQASList(self, interviewResult):
        pass