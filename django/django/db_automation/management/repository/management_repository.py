from abc import ABC, abstractmethod



class ManagementRepository(ABC):

    @abstractmethod
    def userList(self):
        pass


    @abstractmethod
    def grantRoleType(self, email):
        pass


    @abstractmethod
    def revokeRoleType(self, email):
        pass


    @abstractmethod
    def userLogList(self):
        pass


    @abstractmethod
    def userLogData(self):
        pass