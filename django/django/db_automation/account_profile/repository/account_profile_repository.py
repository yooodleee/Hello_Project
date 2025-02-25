from abc import ABC, abstractmethod


class AccountProfileRepository(ABC):

    @abstractmethod
    def save(self, account, nickname):
        pass


    @abstractmethod
    def findByAccount(self, account):
        pass