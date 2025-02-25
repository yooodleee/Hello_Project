from rest_framework import serializers


from company_report.entity.company_report import CompanyReport



class CompanyReportSerializer(serializers.ModelSerializer):

    class Meta:
        model = CompanyReport
        fields = ['companyReportId', 'companyReportName', 'companyReportPrice', 'companyReportTitleImage', 'companyReportCategory', 'keyword', 'content', 'regDate', 'updDate']
        read_only_fields = ['regDate', 'updDate']