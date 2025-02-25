"""
URL configuration for db_automation project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('kakao_oauth/', include('kakao_oauth.urls')),
    path('account/', include('account.urls')),
    path('survey/', include('survey.urls')),
    path('google_oauth/', include('google_oauth.urls')),
    path('naver_oauth/', include('naver_oauth.urls')),
    path('company_report/',include('company_report.urls')),
    path('cart/',include('cart.urls')),
    path('orders/',include('orders.urls')),
    path('marketing/',include('marketing.urls')),
    path('management/',include('management.urls')),
    path('interview/', include('interview.urls')),
    path('interview_result/', include('interview_result.urls'))
]