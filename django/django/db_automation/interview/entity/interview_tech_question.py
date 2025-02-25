from django.db import models



class InterviewTechQuestion(models.Model):
    id = models.AutoField(primary_key=True)
    question = models.CharField(max_length=255)
    job = models.CharField(max_length=64)


    def __str__(self):
        return f"question : {self.question}"
    

    class Meta:
        db_table = 'interview_tech_question'
        app_label = 'interview'