from django.db import models



class Interview(models.Model):
    id = models.AutoField(primary_key=True)
    interview_id = models.IntegerField(null=False)


    def __str__(self):
        return self.interview_id
    

    class Meta:
        db_table = 'interview'
        app_label = 'interview'