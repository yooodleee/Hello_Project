from abc import ABC, abstractmethod


from interview_result.entity.interview_result import InterviewResult
from interview_result.entity.interview_result_qas import InterviewResultQAS
from interview_result.repository.interview_result_repository import InterviewResultRepository



class InterviewResultRepositoryImpl(InterviewResultRepository):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def registerInterviewResult(self, account):
        InterviewResult.objects.create(account=account)
        interviewResult = InterviewResult.objects.all()
        return interviewResult.last()
    

    def registerInterviewResultQAS(self, interviewResult, scoreResultList):
        for scoreResult in scoreResultList:
            print(scoreResult)
            question, answer, intent, feedback = scoreResult
            
            if len(answer) <= 30:
                feedback = '10점<s>답변의 길이가 너무 짧습니다. 질문과 관련한 당신의 구체적인 사례를 언급하여 답변한다면 좋은 점수를 받을 수 있습니다.'
                if any(keyword in answer for keyword in ['모르', '못했', '몰라', '죄송', '모름', '못하']):
                    feedback = '0점<s>답변의 길이가 너무 짧으며, 질문의 의도와 맞지 않습니다. 어려운 질문이라도 최대한 답변할 수 있는 내용을 작성하는 것이 좋습니다.'
            
            InterviewResultQAS.objects.create(
                question=question,
                answer=answer,
                intent=intent,
                feedback=feedback,
                interview_result=interviewResult,
            )
    

    def getLastInterviewResult(self):
        interviewResult = InterviewResult.objects.all()
        return interviewResult.last()
    

    def getLastInterviewResultQASList(self, interviewResult):
        interviewResult = InterviewResultQAS.objects.filter(interview_result=interviewResult)
        interviewResultQASList = interviewResult.order_by('id').values_list('question', 'answer', 'intent', 'feedback')
        
        return interviewResultQASList