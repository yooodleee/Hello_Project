from management.repository.management_repository_impl import ManagementRepositoryImpl
from management.service.management_service import ManagementService



class ManagementServiceImpl(ManagementService):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__managementRepository = ManagementRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def getUserList(self):
        return self.__managementRepository.userList()
    

    def grantRoleType(self, email):
        self.__managementRepository.grantRoleType(email)

    
    def revokeRoleType(self, email):
        self.__managementRepository.revokeRoleType(email)


    def getUserLogList(self):
        return self.__managementRepository.userLogList()
    

    def getUserLogData(self):
        return self.__managementRepository.userLogData()