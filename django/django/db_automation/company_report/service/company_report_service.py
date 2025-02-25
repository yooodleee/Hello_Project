from abc import ABC, abstractmethod



class CompanyReportService(ABC):

    @abstractmethod
    def list(self):
        pass


    @abstractmethod
    def createCompanyReport(self, companyReportName, companyReportPrice, companyReportCategory,content, companyReportTitleImage):
        pass


    @abstractmethod
    def readCompanyReport(self, companyReportId):
        pass


    @abstractmethod
    def deleteCompanyReport(self,companyReportId):
        pass


    @abstractmethod
    def updateCompanyReport(self,companyReportId,companyReportData):
        pass


    @abstractmethod
    def readCompanyReportFinance(self,companyReportName):
        pass


    @abstractmethod
    def readCompanyReportInfo(self,companyReportName):
        pass


    @abstractmethod
    def readTopNCompany(self, topN):
        pass


    @abstractmethod
    def updateCompanyReportDB(self, data):
        pass


    @abstractmethod
    def saveKeyword(self):
        pass