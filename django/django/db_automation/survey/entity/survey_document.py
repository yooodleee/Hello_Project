from django.db import models


from survey.entity.survey import Survey



class SurveyDocument(models.Model):
    id = models.AutoField(primary_key=True)
    survey_id = models.ForeignKey(Survey, on_delete=models.CASCADE, db_column='survey_id')


    class Meta:
        db_table = 'survey_document'
        app_label = 'survey'