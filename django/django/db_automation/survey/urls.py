from django.urls import path, include
from rest_framework.routers import DefaultRouter


from survey.controller.survey_controller import SurveyController


router = DefaultRouter()
router.register(r'survey', SurveyController, basename='survey')


urlpatterns = [
    path('', include(router.urls)),
    path('creat-form', SurveyController.as_view({'post': 'createSurveyForm'}), name='survey-create-form'),
    path('register-title-description', SurveyController.as_view({'post': 'registerTitleDescription'}), name='register-title-description'),
    path('register-question', SurveyController.as_view({'post': 'registerQuestion'}), name='survey-question'),
    path('register-selection', SurveyController.as_view({'post': 'registerSelection'}), name='survey-selection'),
    path('survey-title-list', SurveyController.as_view({'get': 'surveyList'}), name='survey-title-list'),
    path('read-survey-form/<str:randomString>', SurveyController.as_view({'get': 'readSurveyForm'}), name='read-survey-form'),
    path('submit-survey', SurveyController.as_view({'post': 'submitSurvey'}), name='submit-survey'),
    path('randomstring',SurveyController.as_view({'post':'pushRandomstring'}),name='push-randomstring'),
    path('survey-result/<int:surveyId>', SurveyController.as_view({'get': 'surveyResult'}), name='survey-result'),
    path('check-first-submit', SurveyController.as_view({'post': 'checkIsFirstSubmit'}), name='check-first-submit'),
]