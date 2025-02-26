from django.urls import path, include
from rest_framework.routers import DefaultRouter


from naver_oauth.controller.naver_oauth_controller import NaverOauthController


router = DefaultRouter()
router.register(r"naver_oauth", NaverOauthController, basename='naver_oauth')


urlpatterns = [
    path('', include(router.urls)),
    path('naver', NaverOauthController.as_view({'get': 'naverOauthURI'}), name='get-naver-oauth-uri'),
    path('naver/access-token', NaverOauthController.as_view({'post': 'naverAccessTokenURI'}), name='get-naver-access-token-uri'),
    path('naver/user-info', NaverOauthController.as_view({'post': 'naverUserInfoURI'}), name='get-naver-user-info-uri'),
    path('redis-access-token', NaverOauthController.as_view({'post': 'redisAccessToken'}), name='redis_service-access-token'),
    path('logout', NaverOauthController.as_view({'post': 'dropRedisTokenForLogout'}), name='drop-reids_service-token-for-logout'),
]