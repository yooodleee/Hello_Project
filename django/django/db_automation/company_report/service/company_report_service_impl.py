import random


from company_report.repository.company_report_repository_impl import CompanyReportRepositoryImpl
from company_report.service.company_report_service import CompanyReportService



class CompanyReportServiceImpl(CompanyReportService):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__companyReportRepository = CompanyReportRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def list(self):
        return self.__companyReportRepository.list()
    

    def createCompanyReport(self, companyReportName, companyReportPrice, companyReportCategory, content, companyReportTitleImage):
        return self.__companyReportRepository.create(companyReportName, companyReportPrice, companyReportCategory,content, companyReportTitleImage)


    def readCompanyReport(self, companyReportId):
        return self.__companyReportRepository.findByCompanyReportId(companyReportId)


    def deleteCompanyReport(self, companyReportId):
        return self.__companyReportRepository.deleteByCompanyReportId(companyReportId)


    def updateCompanyReport(self, companyReportId, companyReportData):
        companyReport = self.__companyReportRepository.findByCompanyReportId(companyReportId)
        return self.__companyReportRepository.update(companyReport,companyReportData)


    def readCompanyReportFinance(self,companyReportName):
        return self.__companyReportRepository.readCompanyReportFinance(companyReportName)


    def readCompanyReportInfo(self, companyReportName):
        return self.__companyReportRepository.readCompanyReportInfo(companyReportName)


    def readTopNCompany(self, topN):
        return self.__companyReportRepository.readTopNCompany(topN)


    def updateCompanyReportDB(self, data):
        return self.__companyReportRepository.updateDataToDB(data)


    def saveKeyword(self):
        return self.__companyReportRepository.label_and_save_keyword()