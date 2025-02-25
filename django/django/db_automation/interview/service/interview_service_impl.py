import glob
import json
import os


from interview.repository.interview_repository_impl import InterviewRepositoryImpl
from interview.service.interview_serivce import InterviewService



class InterviewServiceImpl(InterviewService):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__interviewRepositoryImpl = InterviewRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def insertSession(self):
        questionFiles = glob.glob(os.path.join('assets/json_qa_pair', '*.json'))
        for questionFile in questionFiles:
            questionList = []
            with open(questionFile, 'r', encoding='utf-8') as file:
                print('열리는 qu file: ', questionFile)
                data = json.load(file)
                for dic in data:
                    questionList.append(dic.get('question'))
            
            interviewId = self.__interviewRepositoryImpl.getMaxId()
            self.__interviewRepositoryImpl.insertData(interviewId+1, questionList)
        print('저장 완료')

        return True
    

    def insertFirstQuestion(self):
        file_path = 'assets\\start_question_3061.json'

        # 파일 열기 및 JSON 파싱
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        [self.__interviewRepositoryImpl.insertFirstQuestion(item['question']) for item in data]

        return True
    

    def insertTechQuestion(self):
        file_path = 'assets\\tech_question_3014.json'

        # 파일 열기 및 JSON 파싱
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        [self.__interviewRepositoryImpl.insertTechQuestion(item['question'], item['job']) for item in data]
        
        return True
    

    def getSession(self, sessionId):
        questionList = self.__interviewRepositoryImpl.getData(sessionId)
        return questionList
    

    def getFirstQuestion(self, questionId):
        firstQuestionList = self.__interviewRepositoryImpl.getFirstQuestion(questionId)
        return firstQuestionList
    

    def getTechQuestion(self, job):
        techQuestion = self.__interviewRepositoryImpl.getTechQuestion(job)
        return techQuestion