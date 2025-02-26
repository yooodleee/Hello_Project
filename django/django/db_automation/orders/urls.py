from django.urls import path, include
from rest_framework.routers import DefaultRouter


from orders.controller.orders_controller import OrdersController


router = DefaultRouter()
router.register(r"orders", OrdersController, basename='orders')


urlpatterns = [
    path('', include(router.urls)),
    path('cart', OrdersController.as_view({'post': 'createCartOrders'}), name='order-cart'),
    path('company_report', OrdersController.as_view({'post': 'createProductOrders'}), name='order-company_report'),
    path('notification', OrdersController.as_view({'post': 'findAccountToNotification'}), name='order-notification'),
    path('list/', OrdersController.as_view({'post': 'myOrderList'}), name='order-list'),
    path('read/<int:pk>', OrdersController.as_view({'post': 'myOrderItemList'}), name='order-item-list'),
    path('order-item-duplication-check', OrdersController.as_view({'post': 'checkOrderItemDuplication'}),
         name='order-item-duplication-check'),
]