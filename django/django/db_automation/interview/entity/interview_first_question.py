from django.db import models



class InterviewFirstQuestion(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=255)

    
    def __str__(self):
        return f"question : {self.question}"
    

    class Meta:
        db_table = 'interview_first_question'
        app_label = 'interview'