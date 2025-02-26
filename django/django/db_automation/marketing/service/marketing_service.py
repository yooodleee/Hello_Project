from abc import ABC, abstractmethod



class MarketingService(ABC):

    @abstractmethod
    def makeCount(self, email, product_id, purchase):
        pass