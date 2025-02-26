from django.db import models
from django.utils import timezone


from account.entity.account import Account
from company_report.entity.company_report import CompanyReport



class Marketing(models.Model):
    id = models.AutoField(primary_key=True)
    account = models.ForeignKey(Account, on_delete=models.CASCADE, related_name='marketing')
    product = models.ForeignKey(CompanyReport, on_delete=models.CASCADE, related_name='marketing')
    click_count = models.PositiveSmallIntegerField(default=1)
    purchase = models.BooleanField(default=False)
    last_click_data = models.DateTimeField(default=timezone.now)


    def __str__(self):
        return f"Marketing -> id: {self.id}, account: {self.account}, product: {self.product}"
    

    class Meta:
        db_table = 'marketing'
        app_label = 'marketing'