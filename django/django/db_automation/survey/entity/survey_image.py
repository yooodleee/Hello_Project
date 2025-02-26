from django.db import models


from survey.entity.survey_question import SurveyQuestion



class SurveyImage(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.TextField(null=True)
    question_id = models.ForeignKey(SurveyQuestion, on_delete=models.CASCADE, db_column='question_id', unique=False)


    class Meta:
        db_table = 'survey_image'
        app_label = 'survey'