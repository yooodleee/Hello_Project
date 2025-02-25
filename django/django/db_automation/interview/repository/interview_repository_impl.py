import random


from interview.entity.interview import Interview
from interview.entity.interview_first_question import InterviewFirstQuestion
from interview.entity.interview_question import InterviewQuestion
from interview.entity.interview_tech_question import InterviewTechQuestion
from interview.repository.interview_repository import InterviewRepository




class InterviewRepositoryImpl(InterviewRepository):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance


    def getMaxId(self):
        interview = Interview.objects.all()
        interviewMaxId = len(interview)
        return interviewMaxId


    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance


    def insertData(self, interviewId, questionList):
        Interview.objects.get_or_create(interview_id=interviewId)
        interview = Interview.objects.get(interview_id=interviewId)
        for question in questionList:
            InterviewQuestion.objects.create(question=question, interview_id=interview)


    def insertFirstQuestion(self, question):
        InterviewFirstQuestion.objects.create(question=question)


    def insertTechQuestion(self, question, job):
        InterviewTechQuestion.objects.create(question=question, job=job)


    def getData(self, sessionId):
        interview = Interview.objects.get(interview_id=sessionId)
        questions = (InterviewQuestion.objects.filter(interview_id=interview).order_by('id').values_list('question'))
        questionList = []
        for question in questions:
            questionList.append(question[0])
        return questionList


    def getFirstQuestion(self, questionId):
        interviewFirstQuestion = InterviewFirstQuestion.objects.get(id=questionId)
        firstQuestion = interviewFirstQuestion.question
        return firstQuestion


    def getTechQuestion(self, job):
        interviewTech = InterviewTechQuestion.objects.filter(job=job)
        interviewTechQuestion = random.choice(interviewTech) if interviewTech else None
        return interviewTechQuestion.question if interviewTechQuestion else None