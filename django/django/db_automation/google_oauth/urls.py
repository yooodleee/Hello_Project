from django.urls import path, include
from rest_framework.routers import DefaultRouter


from google_oauth.controller.google_oauth_controller import GoogleOauthController



router = DefaultRouter()
router.register(r'google_oauth', GoogleOauthController, basename='google_oauth')


urlpatterns = [
    path('', include(router.urls)),
    path('google', GoogleOauthController.as_view({'get': 'googleOauthURI'}), name='get-google-oauth-uri'),
    path('google/access-token', GoogleOauthController.as_view({'post': 'googleAccessTokenURI'}), name='get-google-access-token-uri'),
    path('google/user-info', GoogleOauthController.as_view({'post': 'googleUserInfoURI'}), name='get-google-user-info-uri'),
    path('redis-access-token', GoogleOauthController.as_view({'post': 'redisAccessToken'}), name='redis_service-access-token'),
    path('logout', GoogleOauthController.as_view({'post': 'dropRedisTokenForLogout'}), name='drop-redis_service-token-for-logout')
]