from abc import ABC, abstractmethod



class ManagementService(ABC):

    @abstractmethod
    def getUserList(self):
        pass


    @abstractmethod
    def grantRoleType(self, email):
        pass


    @abstractmethod
    def revokeRoleType(self, email):
        pass


    @abstractmethod
    def getUserLogList(self):
        pass


    @abstractmethod
    def getUserLogData(self):
        pass