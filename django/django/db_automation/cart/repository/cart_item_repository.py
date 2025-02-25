from abc import ABC, abstractmethod


class CartItemRepository(ABC):

    @abstractmethod
    def register(self, cartData, cart, companyReport):
        pass


    @abstractmethod
    def findByCart(self, cart):
        pass


    @abstractmethod
    def findByProductId(self, companyReportId):
        pass


    @abstractmethod
    def findAllByProductId(self, companyReportId):
        pass


    @abstractmethod
    def findById(self, id):
        pass


    @abstractmethod
    def deleteByCartItemId(self, cartItemIdList):
        pass

    
    @abstractmethod
    def checkDuplication(self, cartItemList, companyReportId):
        pass