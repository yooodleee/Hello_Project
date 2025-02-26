from django.db import models


from orders.entity.orders import Orders
from company_report.entity.company_report import CompanyReport



class OrdersItem(models.Model):
    id = models.AutoField(primary_key=True)
    orders = models.ForeignKey(Orders, on_delete=models.CASCADE, related_name='orders_items')
    product = models.ForeignKey(CompanyReport, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)


    def __str__(self):
        return f"OrderItem {self.id} for Order {self.orders.id}"
    
    
    def total_price(self):
        return self.price
    
    
    class Meta:
        db_table = 'orders_item'
        app_label = 'orders'