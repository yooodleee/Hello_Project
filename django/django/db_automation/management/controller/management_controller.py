from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.response import Response


from management.service.management_service_impl import ManagementServiceImpl




class ManagementController(viewsets.ViewSet):
    managementService = ManagementServiceImpl.getInstance()


    def userList(self, request):
        try:
            userList = self.managementService.getUserList()
            return Response({'data': userList}, status=status.HTTP_200_OK)
        except Exception as e:
            print('로그 등록 과정 중 문제 발생: ', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


    def grantRoleType(self, request):
        try:
            email = request.data.get('email')
            self.managementService.grantRoleType(email)
            return Response(status=status.HTTP_200_OK)
        except Exception as e:
            print('권한 변경 과정 중 문제 발생: ', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

    def revokeRoleType(self, request):
        try:
            email = request.data.get('email')
            self.managementService.revokeRoleType(email)
            return Response(status=status.HTTP_200_OK)
        except Exception as e:
            print('권한 변경 과정 중 문제 발생:',e)
            return Response({'error':str(e)},status=status.HTTP_400_BAD_REQUEST)


    def userLogList(self, request):
        try:
            # Service에서 사용자 로그 리스트 데이터를 가져옴
            userLogList = self.managementService.getUserLogList()
            # 성공적으로 데이터를 반환
            return Response({'data': userLogList}, status=status.HTTP_200_OK)
        except Exception as e:
            print('유저 로그 전송 과정 중 문제 발생:', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


    def userLogData(self,request):
        try:
            # Service에서 사용자 로그 리스트 데이터를 가져옴
            userLogData = self.managementService.getUserLogData()
            # 성공적으로 데이터를 반환
            return Response({'data': userLogData}, status=status.HTTP_200_OK)
        except Exception as e:
            print('유저 로그 전송 과정 중 문제 발생:', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)