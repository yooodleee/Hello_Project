from account.repository.account_repository_impl import AccountRepositoryImpl
from interview_result.repository.interview_result_repository_impl import InterviewResultRepositoryImpl
from interview_result.service.interview_result_service import InterviewResultService



class InterviewResultServiceImpl(InterviewResultService):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__interviewResultRepository = InterviewResultRepositoryImpl.getInstance()
            cls.__instance.__accountRepository = AccountRepositoryImpl()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def saveInterviewResult(self, scoreReusltList, accountId):
        account = self.__accountRepository.findById(accountId)
        interviewResult = self.__interviewResultRepository.registerInterviewResult(account)
        self.__interviewResultRepository.registerInterviewResultQAS(interviewResult, scoreReusltList)

    
    def getInterviewResult(self, accountId):
        account = self.__accountRepository.findById(accountId)
        interviewResult = self.__interviewResultRepository.getLastInterviewResult()
        interviewResultList = self.__interviewResultRepository.getLastInterviewResultQASList(interviewResult)

        return interviewResultList