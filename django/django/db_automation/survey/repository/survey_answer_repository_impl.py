from collections import Counter


from survey.entity.survey_answer import SurveyAnswer
from survey.repository.survey_answer_repository import SurveyAnswerRepository



class SurveyAnswerRepositoryImpl(SurveyAnswerRepository):
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
    

    # if accountId is not None:
    #       accountId = self.__accountRepository.findById(accountId)
    #       self.__surveyAnswerRepository.saveAccountId(accountId)
    def saveTextAnswer(self, question, answer, account):
        try:
            if account is not None:
                SurveyAnswer.objects.create(
                    survey_question_id=question,
                    answer=answer,
                    account=account,
                )
            else:
                SurveyAnswer.objects.create(
                    survey_question_id=question,
                    answer=answer,
                )
            return True
        except Exception as e:
            print('text answer 저장 과정에서 오류 발생: ', e)
            return False
    

    def saveRadioAnswer(self, question, selectionId, account):
        try:
            if account is not None:
                SurveyAnswer.objects.create(
                    survey_question_id=question,
                    survey_selection_id=selectionId,
                    account=account,
                )
            else:
                SurveyAnswer.objects.create(
                    survey_question_id=question,
                    survey_selection_id=selectionId,
                )
            return True
        except Exception as e:
            print('radio answer 저장 과정에서 오류 발생: ', e)
            return False
    

    def saveCheckboxAnswer(self, question, selectionIdArray, account):
        try:
            for selectionId in selectionIdArray:
                if account is not None:
                    SurveyAnswer.objects.create(
                        survey_question_id=question,
                        survey_selection_id=selectionId,
                        account=account,
                    )
                else:
                    SurveyAnswer.objects.create(
                        survey_question_id=question,
                        survey_selection_id=selectionId,
                    )
            return True
        except Exception as e:
            print('checkbox answer 저장 과정에서 오류 발생: ', e)
            return False
    

    def getTextAnswerByQuestionId(self, questionId):
        textAnswers = (SurveyAnswer.objects.filter(survey_question_id=questionId).order_by('id').values_list('answer'))
        listTextAnswers = []
        for text in textAnswers:
            listTextAnswers.append(text[0])
        
        return listTextAnswers
    

    def getSelectionAnswerByQuestionId(self, questionId):
        selectionAnswers = (
            SurveyAnswer.objects.filter(survey_question_id=questionId)
            .order_by('id').values_list('survey_selection_id')
        )

        listSelectionAnswers = []
        for s in selectionAnswers:
            listSelectionAnswers.append(s[0])
        
        result = dict(Counter(listSelectionAnswers))
        return result
    

    def getAnswerByAccountId(self, accountId):
        answer = SurveyAnswer.objects.filter(account=accountId)
        if len(answer) != 0:
            return True
        else:
            return False