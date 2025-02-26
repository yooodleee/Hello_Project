from rest_framework import viewsets, status
from rest_framework.response import Response


from survey.service.survey_service_impl import SurveyServiceImpl




class SurveyController(viewsets.ViewSet):
    surveyService = SurveyServiceImpl.getInstance()


    def createSurveyForm(self, request):
        randomString = request.data.get('randomString')
        surveyId = self.surveyService.createSurveyForm(randomString)
        
        return Response(surveyId, status=status.HTTP_200_OK)


    def registerTitleDescription(self, request):
        surveyId = request.data.get('surveyId')
        surveyTitle = request.data.get('surveyTitle')
        surveyDescription = request.data.get('surveyDescription')
        print(f'surveyId : {surveyId}, surveyTitle: {surveyTitle}, surveyDescription: {surveyDescription}')

        survey = self.surveyService.getSurveyBySurveyId(surveyId)
        result = self.surveyService.registerTitleDescription(survey, surveyTitle, surveyDescription)

        return Response(result, status=status.HTTP_200_OK)


    def registerQuestion(self, request):
        surveyId = request.data.get('surveyId')
        questionTitle = request.data.get('questionTitle')
        questionType = request.data.get('questionType')
        essential = request.data.get('isEssential') == 'true'
        images = request.FILES.getlist('images')
        print(
            'surveyId: ', surveyId, 
            'questionTitle: ', questionTitle, 
            'questionType: ', questionType,
            'essential: ', essential, 
            'images: ', 
            images
        )
        survey = self.surveyService.getSurveyBySurveyId(surveyId)
        result = self.surveyService.registerQuestion(survey, questionTitle, questionType, essential, images)

        return Response(result, status=status.HTTP_200_OK)


    def registerSelection(self, request):
        questionId = request.data.get('questionId')
        selection = request.data.get('selection')
        print(f"questionId : {questionId}, selection : {selection}")
        question = self.surveyService.getQuestionByQuestionId(questionId)
        result = self.surveyService.registerSelection(question, selection)

        return Response(result, status=status.HTTP_200_OK)


    def surveyList(self, request):
        surveyTitleList = self.surveyService.getSurveyList()
        randomStringList = self.surveyService.getRandomStringList()
        combinedList = []
        for survey, random in zip(surveyTitleList, randomStringList):
            combinedItem = {**survey, **random}
            combinedList.append(combinedItem)

        return Response({'surveyTitleList': combinedList}, status=status.HTTP_200_OK)


    def readSurveyForm(self, request, randomString=None):
        surveyId = self.surveyService.getSurveyIdByRandomString(randomString)
        surveyForm = self.surveyService.getServeyById(surveyId)
        print('내보낼 결과 : ', surveyForm)

        return Response(surveyForm, status.HTTP_200_OK)


    def submitSurvey(self, request):
        try:
            answers = request.data.get('submitForm')
            accountId = request.data.get('accountId')
            print("answers: ", answers, 'accountId :', accountId)

            self.surveyService.saveAnswer(answers, accountId)

            return Response(True, status.HTTP_200_OK)
        
        except Exception as e:
            return Response(False, status.HTTP_400_BAD_REQUEST)


    def pushRandomstring(self, request):
        try:
            surveyId = self.surveyService.getRecentSurvey()
            data = self.surveyService.getRandomstringBySurveyId(surveyId)
            return Response(data=data,status=status.HTTP_200_OK)
        
        except Exception as e:
            print('randomString 가져오는 중 문제 발생 : ', e)
            return Response(False,status=status.HTTP_400_BAD_REQUEST)


    def surveyResult(self, request, surveyId=None):
        resultForm = self.surveyService.getResultById(surveyId)
        return Response(resultForm, status.HTTP_200_OK)


    def checkIsFirstSubmit(self, request):
        accountId = request.data.get('accountId')
        isSubmitted = self.surveyService.getAnswerByAccountId(accountId)
        return Response(isSubmitted, status.HTTP_200_OK)