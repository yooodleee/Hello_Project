from django.db import models
from survey.entity.survey import Survey



class SurveyDescription(models.Model):
    id = models.AutoField(primary_key=True)
    description = models.CharField(max_length=255, unique=False)
    survey_id = models.OneToOneField(Survey, on_delete=models.CASCADE, db_column='survey_id')


    class Meta:
        db_table = 'survey_description'
        app_label = 'survey'