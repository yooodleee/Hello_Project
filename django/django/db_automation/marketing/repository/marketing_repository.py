from abc import ABC, abstractmethod



class MarketingRepository(ABC):

    @abstractmethod
    def makeCount(self, email, product_id, purchase):
        pass