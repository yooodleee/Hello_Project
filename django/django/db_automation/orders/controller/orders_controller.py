from django.http import JsonResponse
from rest_framework import viewsets, status
from rest_framework.response import Response


from account.repository.profile_repository_impl import ProfileRepositoryImpl
from account.service.account_service_impl import AccountServiceImpl
from orders.repository.orders_item_repository_impl import OrdersItemRepositoryImpl
from orders.service.orders_service_impl import OrdersServiceImpl
from redis_service.service.redis_service_impl import RedisServiceImpl
from company_report.repository.company_report_repository_impl import CompanyReportRepositoryImpl


from datetime import datetime



class OrdersController(viewsets.ViewSet):
    ordersService = OrdersServiceImpl.getInstance()
    redisService = RedisServiceImpl.getInstance()
    accountService = AccountServiceImpl.getInstance()
    productRepository = CompanyReportRepositoryImpl.getInstance()
    profileRepository = ProfileRepositoryImpl.getInstance()
    ordersItemRepository = OrdersItemRepositoryImpl.getInstance()


    def createCartOrders(self, request):
        try:
            data = request.data
            print('data: ', data)

            email = data.get('email')

            accountId = self.profileRepository.findByEmail(email)
            if not accountId:
                raise ValueError('Invalid email')

            account = self.accountService.findAccountById(accountId.account_id)

            orderItemList = data.get('items')
            print(f"orderItemList: {orderItemList}")

            orderId = self.ordersService.createCartOrder(account, orderItemList)
            return Response(orderId, status=status.HTTP_200_OK)
        
        except Exception as e:
            print("주문 과정 중 문제 발생: ", e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

    def createProductOrders(self, request):
        try:
            data = request.data
            print('data: ', data)

            email = data.get('email')
            accountId = self.profileRepository.findByEmail(email)
            if not accountId:
                raise ValueError('Invalid email')
            
            account = self.accountService.findAccountById(accountId.account_id)
            companyReportId = data.get('companyReportId')
            companyReport = self.productRepository.findByCompanyReportId(companyReportId)
            companyReportPrice = data.get('companyReportPrice')
            quantity = 1

            orderItem = {
                "company_report": companyReport,
                "companyReportPrice": companyReportPrice,
                "quantity": quantity,
            }
            orderId = self.ordersService.createProductOrder(account, orderItem)
            return Response(orderId, status=status.HTTP_200_OK)
        
        except Exception as e:
            print("주문 과정 중 문제 발생: ", e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

    def myOrderList(self, request):
        email = request.data.get('email')
        print('email: ', email)
        accountId = self.profileRepository.findByEmail(email)
        ordersList = self.ordersService.findAllByAccountId(accountId.account_id)
        serializedOrdersList = []

        for orders in ordersList:
            totalPrice = 0
            ordersItemList = self.ordersItemRepository.findAllByOrdersId(orders.id)
            for ordersItem in ordersItemList:
                totalPrice += ordersItem.price
            
            serializedOrdersList.append(
                {
                    'ordersId': orders.id,
                    'createdDate': orders.createdDate,
                    'totalPrice': totalPrice,
                    'totalQuantity': len(ordersItemList),
                }
            )
        
        return JsonResponse(serializedOrdersList, safe=False, status=status.HTTP_200_OK)
    

    def myOrderItemList(self, request, pk=None):
        ordersItemList = self.ordersItemRepository.findAllByOrdersId(pk)
        serializedOrdersItemList = [
            {
                'companyReportId': ordersItem.product.companyReportId,
                'companyReportTitleImage': ordersItem.product.companyReportTitleImage,
                'companyReportName': ordersItem.product.companyReportName,
                'companyReportPrice': ordersItem.product.companyReportPrice,
            }
            for ordersItem in ordersItemList
        ]

        return JsonResponse(serializedOrdersItemList, safe=False, status=status.HTTP_200_OK)
    

    def checkOrderItemDuplication(self, request):
        email = request.data['payload']['email']
        companyReportId = int(request.data['payload']['companyReportId'])

        accountId = self.profileRepository.findByEmail(email)
        ordersList = self.ordersService.findAllByAccountId(accountId.account_id)
        ordersIdList = [orders.id for orders in ordersList]
        allOrdersItemList = [self.ordersItemRepository.findAllByOrdersId(ordersId) for ordersId in ordersIdList]
        isDuplicate = self.ordersItemRepository.checkDuplication(allOrdersItemList, companyReportId)
        print(f"isDuplicate: {isDuplicate}")

        return JsonResponse(isDuplicate, status=status.HTTP_200_OK)