from django.db import models


from survey.entity.survey import Survey



class SurveyTitle(models.Model):
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=64, unique=False)
    survey_id = models.ForeignKey(Survey, on_delete=models.CASCADE, db_column='survey_id')


    class Meta:
        db_table = 'survey_title'
        app_label = 'survey'