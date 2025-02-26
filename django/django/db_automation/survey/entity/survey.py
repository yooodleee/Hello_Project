from django.db import models



class Survey(models.Model):
    id = models.AutoField(primary_key=True)
    survey = models.CharField(max_length=128)


    def __str__(self):
        return f"Survey -> id: {self.id}"

    
    class Meta:
        db_table = 'survey'
        app_label = 'survey'