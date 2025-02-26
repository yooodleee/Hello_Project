from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import viewsets, status


from db_automation.redis_service.service.redis_service_impl import RedisCacheServiceImpl



class AuthenticationController(viewsets.ViewSet):
    redisCacheService = RedisCacheServiceImpl.getInstance()


    def requestLogout(self, request):
        postRequest = request.data
        userToken = postRequest.get("userToken")

        if userToken:
            try:
                accountId = self.redisCacheService.getValueByKey(userToken)
                self.redisCacheService.deleteKey(userToken)
                self.redisCacheService.deleteKey(accountId)
                return JsonResponse(
                    {"message": "로그 아웃 성공"}, status=status.HTTP_200_OK
                )
            except Exception as e:
                print(f"redis key 삭제 중 에러 발생: {e}")
                return JsonResponse({"error": "코드 내부 에러"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return JsonResponse({"error": "userToken이 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)
    

    def requestUserTokenValidation(self, request):
        postRequest = request.data
        userToken = postRequest.get("userToken")

        if not userToken:
            return JsonResponse({"valid": False, "error": "userToken이 필요합니다."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            accountId = self.redisCacheService.getValueByKey(userToken)
            if not accountId:
                return JsonResponse({"valid": False}, status=status.HTTP_200_OK)
            
            return JsonResponse({"valid": True}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return JsonResponse({"valid": False, "error": "코드 내부 에러"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)