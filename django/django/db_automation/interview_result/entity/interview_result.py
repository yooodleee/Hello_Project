from django.db import models

from account.entity.account import Account



class InterviewResult(models.Model):
    id = models.AutoField(primary_key=True)
    account = models.ForeignKey(Account, on_delete=models.CASCADE, unique=False)


    def __str__(self):
        return f"InterviewResult -> id: {self.id}"
    

    class Meta:
        db_table = 'interview_result'
        app_label = 'interview_result'