from survey.entity.survey_description import SurveyDescription
from survey.repository.survey_description_repository import SurveyDescriptionRepository



class SurveyDescriptionRepositoryImpl(SurveyDescriptionRepository):
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
    

    def registerDescription(self, survey, surveyDescription):
        try:
            SurveyDescription.objects.create(
                survey_id=survey,
                description=surveyDescription,
            )
            return True
        except Exception as e:
            print('Description 저장 중 오류 발생: ', e)
            return True
    

    def getDescriptionBySurveyId(self, surveyId):
        description = SurveyDescription.objects.get(survey_id=surveyId)
        return description.description