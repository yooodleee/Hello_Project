from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import viewsets, status


from account.service.account_service_impl import AccountServiceImpl
from db_automation.redis_service.service.redis_service_impl import RedisServiceImpl


class AccountController(viewsets.ViewSet):
    __accountService = AccountServiceImpl.getInstance()
    redisCacheService = RedisServiceImpl.getInstance()

    def requestEmail(self, request):
        postRequest = request.data
        userToken = postRequest.get("userToken")

        # userToken이 없으면 400 오류 반환
        if not userToken:
            return JsonResponse(
                {"error": "userToken이 필요합니다", "success": False}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Redis에서 userToken에 해당하는 accountId를 가져옴
            accountId = self.redisCacheService.getValueByKey(userToken)

            if not accountId:
                # Redis에서 accountId를 찾지 못한 경우
                return JsonResponse({"error": "유효한 userToken이 아닙니다", "success": False}, status=status.HTTP_404_NOT_FOUND)

            # accountId를 사용하여 이메일을 찾음
            foundEmail = self.__accountService.findEmail(accountId)

            if foundEmail is None:
                # 이메일을 찾지 못한 경우
                return JsonResponse({"error": "이메일을 찾을 수 없습니다", "success": False}, status=status.HTTP_404_NOT_FOUND)

            # 이메일을 찾았으면 200 OK 응답
            return JsonResponse({"email": foundEmail, "success": True}, status=status.HTTP_200_OK)

        except Exception as e:
            # 예외 처리
            print(f"서버 오류 발생: {e}")
            return JsonResponse({"error": "서버 내부 오류", "success": False}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)