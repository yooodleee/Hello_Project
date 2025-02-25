from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response


from account.repository.profile_repository_impl import ProfileRepositoryImpl
from account.service.account_service_impl import AccountServiceImpl
from cart.entity.cart_item import CartItem
from cart.repository.cart_item_repository_impl import CartItemRepositoryImpl
from cart.repository.cart_repository_impl import CartRepositoryImpl
from cart.service.cart_service_impl import CartServiceImpl



class CartController(viewsets.ViewSet):
    cartService = CartServiceImpl.getInstance()
    cartRepository = CartRepositoryImpl.getInstance()
    cartItemRepository = CartItemRepositoryImpl.getInstance()
    profileRepository = ProfileRepositoryImpl.getInstance()
    accountService = AccountServiceImpl.getInstance()


    def cartItemList(self, request):
        data = request.data
        email = data.get('email')

        if not email:
            return Response({'error': 'User token is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        accountId = self.profileRepository.findByEmail(email)
        print(accountId.account_id)
        if not accountId:
            return Response({'error': 'Invalid user token'}, status=status.HTTP_400_BAD_REQUEST)
        
        cartItemListReponseForm = self.cartService.cartList(accountId.account_id)
        return Response(cartItemListReponseForm, status=status.HTTP_200_OK)
    

    def cartRegister(self, request):
        try:
            data = request.data
            print('data: ', data)

            email = data.get('email')
            accountId = self.profileRepository.findByEmail(email)

            self.cartService.cartRegister(data, accountId.account_id)
            return Response(status=status.HTTP_200_OK)
        
        except Exception as e:
            print('상품 등록 과정 중 문제 발생: ', e)
            return Response({ 'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

    def removeCartItem(self, request):
        data = request.data

        if list(data.keys())[0] == 'companyReportId':
            cartItemList = CartItem.objects.all()
            for cartItem in cartItemList:
                if cartItem.product.companyReportId == int(data['companyReportId'][0]):
                    self.cartService.removeCartItem([cartItem.cartItemId])
        
        if list(data.keys())[0] == 'CartItemId':
            self.cartService.removeCartItem(data['CartItemId'])
        
        return Response(status=status.HTTP_204_NO_CONTENT)
    

    def checkCartItemDuplication(self, request):
        email = request.data['payload']['email']
        companyReportId = request.data['payload']['companyReportId']

        accountId = self.profileRepository.findByEmail(email)
        account = self.accountService.findAccountById(accountId.account_id)
        cart = self.cartRepository.findByAccount(account)
        cartItemList = self.cartItemRepository.findByCart(cart)
        isDuplicate = self.cartItemRepository.checkDuplication(cartItemList, companyReportId)
        
        print(f"isDuplicate: {isDuplicate}")
        return Response(isDuplicate, status=status.HTTP_200_OK)