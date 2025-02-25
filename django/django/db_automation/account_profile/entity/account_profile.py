from django.db import models


from account.entity.account import Account



class AccountProfile(models.Model):
    id = models.AutoField(primary_key=True)
    nickname = models.CharField(max_length=32, unique=True)
    account = models.OneToOneField(
        Account,
        on_delete=models.CASCADE,
        related_name='profile',
    )

    class Meta:
        db_table = 'account_profile'
        app_label = 'account_profile'