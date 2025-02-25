from django.urls import path, include
from rest_framework.routers import DefaultRouter


from cart.controller.cart_controller import CartController


router = DefaultRouter()
router.register(r"cart", CartController, basename="cart")


urlpatterns = [
    path('', include(router.urls)),
    path('list', CartController.as_view({'post': 'cartItemList'}), name='cart-list'),
    path('register', CartController.as_view({'post': 'cartRegister'}), name='cart-register'),
    path('delete', CartController.as_view({'delete': 'removeCartItem'}), name='cartItem-remove'),
    path('cart-item-duplication-check', CartController.as_view({'post': 'checkCartItemDuplication'}), 
         name='cartItem-duplication-check'),
]