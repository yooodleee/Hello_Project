from abc import ABC, abstractmethod



class OrdersRepository(ABC):

    @abstractmethod
    def create(self, account):
        pass


    @abstractmethod
    def getAllOrders(self):
        pass


    @abstractmethod
    def findAllByAccount(self, accountId):
        pass