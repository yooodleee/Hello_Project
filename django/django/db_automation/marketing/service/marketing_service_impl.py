from marketing.repository.marketing_repository_impl import MarketingRepositoryImpl
from marketing.service.marketing_service import MarketingService



class MarketingServiceImpl(MarketingService):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__marketingRepository = MarketingRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def makeCount(self, email, product_id, purchase):
        return self.__marketingRepository.makeCount(email, product_id, purchase)