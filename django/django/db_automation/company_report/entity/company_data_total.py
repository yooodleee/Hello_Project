from django.db import models



class CompanyDataTotal(models.Model):
    id = models.AutoField(primary_key=True)
    company_name = models.CharField(max_length=128, null=False)
    est_date = models.CharField(max_length=20, null=True)
    company_class = models.CharField(max_length=20, null=True)
    ceo_name = models.CharField(max_length=50, null=True)
    address = models.CharField(max_length=255, null=True)
    website = models.URLField(max_length=200, null=True)
    business_summary = models.TextField(null=True)
    revenue_table = models.TextField(null=True)


    def __str__(self):
        return self.company_name


    class Meta:
        db_table = 'company_data'
        app_label = 'company_report'