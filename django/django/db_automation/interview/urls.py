from django.urls import path, include
from rest_framework.routers import DefaultRouter


from interview.controller.interview_controller import InterviewController

router = DefaultRouter()
router.register(r'interview', InterviewController, basename='interview')
urlpatterns = [
    path('', include(router.urls)),
    path('insert-session', InterviewController.as_view({'post': 'insertSession'}), name='insert-session'),
    path('insert-first-question', InterviewController.as_view({'post': 'insertFirstQuestion'}), name='insert-first-question'),
    path('insert-tech-question', InterviewController.as_view({'post': 'insertTechQuestion'}), name='insert-tech-question'),
    path('get-session', InterviewController.as_view({'post': 'getSession'}), name='get-session'),
    path('get-first-question', InterviewController.as_view({'post': 'getFirstQuestion'}), name='get-first-question'),
    path('get-tech-question', InterviewController.as_view({'post': 'getTechQuestion'}), name='get-tech-question'),
]