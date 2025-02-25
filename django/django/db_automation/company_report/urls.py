from django.urls import path, include
from rest_framework.routers import DefaultRouter

from company_report.controller.company_report_controller import CompanyReportController

router = DefaultRouter()
router.register(r'company_report', CompanyReportController)

urlpatterns = [
    path('', include(router.urls)),
    path('list/', CompanyReportController.as_view({'get': 'list'}), name='company_report-list'),
    path('register', CompanyReportController.as_view({'post': 'register'}), name='company_report-register'),
    path('read/<int:pk>', CompanyReportController.as_view({'get': 'readCompanyReport'}), name='company_report-read'),
    path('delete/<int:pk>',CompanyReportController.as_view({'delete': 'deleteCompanyReport'}),name='company_report-delete'),
    path('modify/<int:pk>', CompanyReportController.as_view({'put': 'modifyCompanyReport'}), name='company_report-modify'),
    path('finance',CompanyReportController.as_view({'post':'readCompanyReportFinance'}),name='company-report-finance'),
    path('info',CompanyReportController.as_view({'post':'readCompanyReportInfo'}),name='company-report-info'),
    path('top',CompanyReportController.as_view({'post':'readTopClickedCompany'}),name='company-report-top'),
    path('update',CompanyReportController.as_view({'post':'updateReport'}),name='company-report-update'),
    path('keyword',CompanyReportController.as_view({'post':'saveKeyword'}),name='company-report-keyword')
]