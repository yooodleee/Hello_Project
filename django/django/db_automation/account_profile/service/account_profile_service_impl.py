from account.repository.account_repository_impl import AccountRepositoryImpl
from account_profile.repository.account_profile_repository_impl import AccountProfileRepositoryImpl
from account_profile.service.account_profile_service import AccountProfileService



class AccountProfileServiceImpl(AccountProfileService):
    __instance = None

    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

            cls.__instance.__accountProfileRepository = AccountProfileRepositoryImpl.getInstance()
            cls.__instance.__accountRepository = AccountRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def createAccountProfile(self, accountId, nickname):
        if not nickname:
            nickname = "temporary"
        
        account = self.__accountRepository.findById(accountId)
        savedAccountProfile = self.__accountProfileRepository.save(account, nickname)
        if savedAccountProfile is not None:
            return True
        
        return False