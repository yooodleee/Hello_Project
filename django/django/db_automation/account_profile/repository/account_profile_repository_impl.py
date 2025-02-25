from django.db import IntegrityError


from account_profile.entity.account_profile import AccountProfile
from account_profile.repository.account_profile_repository import AccountProfileRepository


class AccountProfileRepositoryImpl(AccountProfileRepository):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__itemsize__ = cls()
        
        return cls.__instance
    

    def save(self, account, nickname):
        try:
            accountProfile = AccountProfile.objects.create(account=account, nickname=nickname)
            return accountProfile
        
        except IntegrityError:
            raise IntegrityError(f"Nickname '{nickname}' 이미 존재함.")
    

    def findByAccount(self, account):
        try:
            # 주어진 Account 객체에 해당하는 AccountProfile을 조회
            return AccountProfile.objects.get(account=account)
        
        except AccountProfile.DoesNotExist:
            # 만약 해당하는 AccountProfile이 없으면 None 반환
            return None