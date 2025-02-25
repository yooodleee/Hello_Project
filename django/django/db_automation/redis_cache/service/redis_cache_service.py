from abc import ABC, abstractmethod


class RedisCacheService(ABC):

    @abstractmethod
    def storeKeyValue(self, key, value):
        pass


    @abstractmethod
    def getValueByKey(self, key):
        pass


    @abstractmethod
    def deleteKey(self, key):
        pass