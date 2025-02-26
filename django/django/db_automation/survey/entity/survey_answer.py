from django.db import models


from account.entity.account import Account
from survey.entity.survey_question import SurveyQuestion
from survey.entity.survey_selection import SurveySelection



class SurveyAnswer(models.Model):
    id = models.AutoField(primary_key=True)
    answer = models.CharField(max_length=128, null=True, unique=False)
    survey_selection_id = models.ForeignKey(SurveySelection, on_delete=models.CASCADE, db_column='survey_selection_id', null=True)
    survey_question_id = models.ForeignKey(SurveyQuestion, on_delete=models.CASCADE, db_column='survey_question_id', null=False)
    account = models.ForeignKey(Account, on_delete=models.CASCADE, db_column='account_id', null=True, default=None) # 로그인 안한 유저일 시 null 값이 들어감


    class Meta:
        db_table = 'survey_answer'
        app_label = 'survey'