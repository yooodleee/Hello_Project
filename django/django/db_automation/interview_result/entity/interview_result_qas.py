from django.db import models

from interview_result.entity.interview_result import InterviewResult



class InterviewResultQAS(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=200, unique=False)
    answer = models.TextField(unique=False)
    intent = models.CharField(max_length=8, unique=False)
    feedback = models.TextField(unique=False)
    interview_result = models.ForeignKey(InterviewResult, on_delete=models.CASCADE, db_column='interview_result')


    class Meta:
        db_table = 'interview_result_qa_score'
        app_label = 'interview_result'