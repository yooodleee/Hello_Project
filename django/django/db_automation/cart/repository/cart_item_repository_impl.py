from cart.entity.cart import Cart
from cart.entity.cart_item import CartItem
from cart.repository.cart_item_repository import CartItemRepository



class CartItemRepositoryImpl(CartItemRepository):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def register(self, cartData, cart, companyReport):
        companyReportPrice = cartData.get('companyReportPrice')

        CartItem.objects.create(
            cart=cart,
            product=companyReport,
            quantity=1,
            price=companyReportPrice,
        )
    

    def findByCart(self, cart):
        return list(CartItem.objects.filter(cart=cart))
    

    def findByProductId(self, companyReportId):
        try:
            return CartItem.objects.get(product_id=companyReportId)
        except CartItem.DoesNotExist:
            return None
    

    def findAllByProductId(self, companyReportId):
        return CartItem.objects.filter(product_id=companyReportId)
    

    def findById(self, id):
        return CartItem.objects.get(cartItemId=id)
    

    def deleteByCartItemId(self, cartItemIdList):
        for cartItemId in cartItemIdList:
            cartItem = CartItem.objects.get(cartItemId=cartItemId)
            cartItem.delete()
    

    def checkDuplication(self, cartItemList, companyReportId):
        for cartItem in cartItemList:
            if str(cartItem.product.companyReportId) == str(companyReportId):
                return True
        
        return False