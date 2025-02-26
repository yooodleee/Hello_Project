from account.repository.account_repository_impl import AccountRepositoryImpl
from survey.repository.survey_answer_repository_impl import SurveyAnswerRepositoryImpl
from survey.repository.survey_description_repository_impl import SurveyDescriptionRepositoryImpl
from survey.repository.survey_question_repository_impl import SurveyQuestionRepositoryImpl
from survey.repository.survey_repository_impl import SurveyRepositoryImpl
from survey.repository.survey_selection_repository_impl import SurveySelectionRepositoryImpl
from survey.repository.survey_title_repository_impl import SurveyTitleRepositoryImpl
from survey.service.survey_service import SurveyService



class SurveyServiceImpl(SurveyService):
    __instance = None

    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            # cls.__instance.__surveyDocumentRepository = SurveyDocumentRepositoryImpl.getInstance()
            cls.__instance.__surveyRepository = SurveyRepositoryImpl.getInstance()
            cls.__instance.__surveyTitleRepository = SurveyTitleRepositoryImpl.getInstance()
            cls.__instance.__surveyDescriptionRepository = SurveyDescriptionRepositoryImpl.getInstance()
            cls.__instance.__surveyQuestionRepository = SurveyQuestionRepositoryImpl.getInstance()
            cls.__instance.__surveySelectionRepository = SurveySelectionRepositoryImpl.getInstance()
            cls.__instance.__surveyAnswerRepository = SurveyAnswerRepositoryImpl.getInstance()
            cls.__instance.__accountRepository = AccountRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def getRecentSurvey(self):
        recentId = self.__surveyRepository.getMaxId()
        return recentId
    

    def createSurveyForm(self, randomString):
        maxId = self.__surveyRepository.getMaxId()
        self.__surveyRepository.registerSurvey(randomString)
        return maxId + 1
    

    def getSurveyBySurveyId(self, surveyId):
        survey = self.__surveyRepository.findSurvey(surveyId)
        return survey
    

    def getQuestionByQuestionId(self, questionId):
        question = self.__surveyQuestionRepository.findQuestion(questionId)
        return question
    

    def registerTitleDescription(self, survey, surveyTitle, surveyDescription):
        try:
            titleResult = self.__surveyTitleRepository.registerTitle(survey, surveyTitle)
            descriptionResult = self.__surveyDescriptionRepository.registerDescription(survey, surveyDescription)
            return titleResult & descriptionResult
        except Exception as e:
            print('설문 제목, 설명 저장 중 오류 발생: ', e)
            return False
    

    def registerQuestion(self, survey, questionTitle, questionType, essential, images):
        try:
            question = (
                self.__surveyQuestionRepository.registerQuestion(survey, questionTitle, questionType, essential, images)
            )
            return question
        except Exception as e:
            print('설문 제목 저장 중 오류 발생: ', e)
            return False
    

    def registerSelection(self, question, selection):
        try:
            result = self.__surveySelectionRepository.registerSelection(question, selection)
            return result
        except Exception as e:
            print('설문 선택 항목 저장 중 오류 발생: ', e)
            return False
    

    def getSurveyList(self):
        return self.__surveyTitleRepository.getAllTitles()
    

    def getRandomStringList(self):
        return self.__surveyRepository.getAllRandomString()
    

    def getServeyById(self, surveyId):
        surveyTitle = self.__surveyTitleRepository.getTitleBySurveyId(surveyId)
        surveyDescription = self.__surveyDescriptionRepository.getDescriptionBySurveyId(surveyId)
        surveyQuestions = self.__surveyQuestionRepository.getQuestionBySurveyId(surveyId)

        for question in surveyQuestions:
            if question['questionType'] != 'text':
                selection = self.__surveySelectionRepository.getSelectionByQuestionId(question['questionId'])
                question['selection'] = selection
        
        surveyForm = {
            'surveyId': surveyId,
            'surveyTitle': surveyTitle,
            'surveyDescription': surveyDescription,
            'surveyQuestions': surveyQuestions,
        }

        return surveyForm
    

    def saveAnswer(self, answers, account):
        try:
            if account is not None:
                account = self.__accountRepository.findById(account)
            
            for answer in answers:
                questionId = answers.get('questionId')
                question = self.__surveyQuestionRepository.findQuestion(questionId)

                if answer['questionType'] == 'text':
                    answer = answer.get('answer')
                    textAnswer = self.__surveyAnswerRepository.saveTextAnswer(question, answer, account)
                
                elif answer['questionType'] == 'radio':
                    selection = answer.get('answer')
                    selection = self.__surveySelectionRepository.findSelectionBySelectionName(question, selection)
                    checkboxAnswer = self.__surveyAnswerRepository.saveRadioAnswer(question, selection, account)
                
                elif answer['questionType'] == 'checkbox':
                    selectionNameArray = answer.get('answer')
                    selectionArray = [
                        self.__surveySelectionRepository.findSelectionBySelectionName(question, selection)
                        for selection in selectionNameArray
                    ]
                    radioAnswer = self.__surveyAnswerRepository.saveCheckboxAnswer(question, selectionArray, account)
        
        except Exception as e:
            print('답변 저장 중 오류 발생: ', e)
    

    def getSurveyIdByRandomString(self, randomString):
        return self.__surveyRepository.findSurveyIdByRandomString(randomString)
    

    def getRandomstringBySurveyId(self, surveyId):
        return self.__surveyRepository.findRandomStringBySurveyId(surveyId)
    

    def getResultById(self, surveyId):
        surveyTitle = self.__surveyTitleRepository.getTitleBySurveyId(surveyId)
        surveyDescription = self.__surveyDescriptionRepository.getDescriptionBySurveyId(surveyId)
        surveyQuestions = self.__surveyQuestionRepository.getQuestionBySurveyId(surveyId)

        for question in surveyQuestions:
            if question['questionType'] == 'text':
                answer = self.__surveyAnswerRepository.getTextAnswerByQuestionId(question['questionId'])
                question['answer'] = answer
            else:
                selectionAnswer = self.__surveyAnswerRepository.getSelectionAnswersByQuestionId(question['questionId'])
                convertedData = {}
                for selectionId, value in selectionAnswer.items():
                    selectionName = self.__surveySelectionRepository.findSelectionBySelectionId(selectionId).selection
                    convertedData[selectionName] = value

                question['selection'] = convertedData
        resultForm = {
            'surveyId': surveyId, 
            'surveyTitle': surveyTitle,
            'surveyDescription': surveyDescription, 
            'surveyQuestions': surveyQuestions
        }
        
        print('resultForm 생성이 완료되었습니다 : \n', resultForm)
        return resultForm


    def getAnswerByAccountId(self, accountId):
        accountId = self.__accountRepository.findById(accountId)
        isSubmitted = self.__surveyAnswerRepository.getAnswerByAccountId(accountId)
        return isSubmitted