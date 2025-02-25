from abc import ABC, abstractmethod


class AccountProfileService(ABC):

    @abstractmethod
    def createAccountProfile(self, accountId, nickname):
        pass