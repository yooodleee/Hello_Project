from django.urls import path, include
from rest_framework.routers import DefaultRouter


from marketing.controller.marketing_controller import MarketingController


router = DefaultRouter()
router.register(r"marketing", MarketingController, basename='marketing')


urlpatterns = [
    path('', include(router.urls)),
    path('make-count', MarketingController.as_view({'post': 'clickCount'}), name='click_count'),
]