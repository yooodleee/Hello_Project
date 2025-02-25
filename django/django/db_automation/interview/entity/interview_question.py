from django.db import models
from interview.entity.interview import Interview



class InterviewQuestion(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=255)
    interview_id = models.ForeignKey(Interview, on_delete=models.CASCADE, db_column='interview_id')
    

    def __str__(self):
        return f"question : {self.question}"
    

    class Meta:
        db_table = 'interview_question'
        app_label = 'interview'