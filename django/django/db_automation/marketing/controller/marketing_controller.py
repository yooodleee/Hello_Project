from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.response import Response


from marketing.service.marketing_service_impl import MarketingServiceImpl



class MarketingController(viewsets.ViewSet):
    marketingService = MarketingServiceImpl.getInstance()


    def clickCount(self, request):
        try:
            email = request.data.get('email')
            companyReport_id = request.data.get('companyReport_id')
            purchase = request.data.get('purchase') # Boolean 값으로 직접 사용
            print(purchase, email, companyReport_id)

            if purchase is not None:
                marketing_instance = self.marketingService.makeCount(email, companyReport_id, purchase)

                return Response(status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Purchase value is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            print('로그 등록 과정 중 문제 발생: ', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)