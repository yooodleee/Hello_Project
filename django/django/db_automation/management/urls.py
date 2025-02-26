from django.urls import path, include
from rest_framework.routers import DefaultRouter


from management.controller.management_controller import ManagementController


router = DefaultRouter()
router.register(r"management", ManagementController, basename='management')


urlpattenrs = [
    path('', include(router.urls)),
    path('userList', ManagementController.as_view({'get': 'userList'}), name='user-list'),
    path('grant-roleType', ManagementController.as_view({'post': 'grantRoleType'}), name='account-grant-role-type'),
    path('revoke-roleType', ManagementController.as_view({'post': 'revokeRoleType'}), name='account-revoke-role-type'),
    path('userLogList', ManagementController.as_view({'post': 'userLogList'}), name='user-log-list'),
    path('data', ManagementController.as_view({'post': 'userLogData'}), name='user-log-data'),
]