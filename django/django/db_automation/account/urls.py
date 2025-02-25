from tkinter.font import names
from django.urls import path, include
from rest_framework.routers import DefaultRouter


from account.controller.account_controller import AccountController



router = DefaultRouter()
router.register(r'account', AccountController, basename='account')


urlpatterns = [
    path('', include(router.urls)),
    path('get-account-id', AccountController.as_view({'post': 'getAccountId'}), name='get-account-id'),
    path('email-duplication-check',
         AccountController.as_view({'post': 'checkEmailDuplication'}), name='account-email-duplication-check'),
    path('nickname-duplication-check',
         AccountController.as_view({'post': 'checkNicknameDuplication'}), name='account-nickname-duplication-check'),
    path('register', AccountController.as_view({'post': 'registerAccount'}), name='register-account'),
    path('nickname', AccountController.as_view({'post': 'getNickname'}), name='nickname-account'),
    path('email', AccountController.as_view({'post': 'getEmail'}), name='email-account'),
    path('withdraw', AccountController.as_view({'post': 'withdrawAccount'}), name='withdraw-account'),
    path('gender', AccountController.as_view({'post': 'getGender'}), name='gender-account'),
    path('birthyear', AccountController.as_view({'post': 'getBirthyear'}), name='birthyear-account'),
    path('account-check', AccountController.as_view({'post': 'checkPassword'}), name='normal-login-check-account'),
    path('modify-nickname',AccountController.as_view({'post':'modifyNickname'}),name='account-modify-nickname'),
    path('modify-password',AccountController.as_view({'post':'modifyPassword'}),name='account-modify-password'),
    path('role-type',AccountController.as_view({'post':'getRoleType'}),name='account-role-type'),
    path('profile',AccountController.as_view({'post':'getProfile'}),name='account-profile'),
]