from django.urls import path, include
from rest_framework.routers import DefaultRouter

from kakao_oauth.controller.kakao_oauth_controller import KakaoOauthController

router = DefaultRouter()
router.register(r'kakao_oauth', KakaoOauthController, basename='kakao_oauth')

urlpatterns = [
    path('', include(router.urls)),
    path('kakao', KakaoOauthController.as_view({'get': 'kakaoOauthURI'}), name='get-kakao-oauth-uri'),
    path('kakao/access-token', KakaoOauthController.as_view({'post': 'kakaoAccessTokenURI'}), name='get-kakao-access-token-uri'),
    path('kakao/user-info', KakaoOauthController.as_view({'post': 'kakaoUserInfoURI'}),
                                name='get-kakao-user-info-uri'),
    path('redis-access-token/', KakaoOauthController.as_view({'post': 'redisAccessToken'}), name='redis_service-access-token'),
    path('logout', KakaoOauthController.as_view({'post': 'dropRedisTokenForLogout'}), name='drop-redis_service-token-for-logout')
]