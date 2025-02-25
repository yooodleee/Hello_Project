import json
import os
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response


from company_report.entity.company_report import CompanyReport
from company_report.serializers import CompanyReportSerializer
from company_report.service.company_report_service_impl import CompanyReportServiceImpl



class CompanyReportController(viewsets.ViewSet):
    queryset = CompanyReport.objects.all()
    companyReportService = CompanyReportServiceImpl.getInstance()


    def list(self, request):
        companyReportList = self.companyReportService.list()
        # print('companyReportList : ', companyReportList)
        serializer = CompanyReportSerializer(companyReportList, many=True)
        
        return Response(serializer.data)
    

    def register(self, request):
        try:
            data = request.data
            companyReportTitleImage = request.FILES.get('companyReportTitleImage')
            companyReportName = data.get('companyReportName')
            companyReportPrice = data.get('companyReportPrice')
            companyReportCategory = data.get('companyReportCategory')
            content = data.get('content')

            if not all([companyReportName, companyReportPrice, companyReportCategory, content, companyReportTitleImage]):
                return Response(
                    {'error': '모든 내용을 채워주세요!'}, status=status.HTTP_400_BAD_REQUEST
                )
            
            self.companyReportService.createCompanyReport(
                companyReportName, companyReportPrice, companyReportCategory, content, companyReportTitleImage
            )

            serializer = CompanyReportSerializer(data=request.data)
            return Response(status=status.HTTP_200_OK)
        
        except Exception as e:
            print('상품 등록 과정 중 문제 발생: ', e)
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

    def readCompanyReport(self, request, pk=None):
        companyReport = self.companyReportService.readCompanyReport(pk)
        serializer = CompanyReportSerializer(companyReport)
        return Response(serializer.data)
    

    def deleteCompanyReport(self, request, pk=None):
        self.companyReportService.deleteCompanyReport(pk)
        return Response(status=status.HTTP_204_NO_CONTENT)
    

    def modifyCompanyReport(self, request, pk=None):
        companyReport = self.companyReportService.readCompanyReport(pk)
        # print(f"request Data: {request.data}")
        serializer = CompanyReportSerializer(companyReport, data=request.data, partial=True)
        # print(f"Request Data: {request.data}")  # 이 부분에서 front에서 전달된 데이터를 확인

        if serializer.is_valid():
            # print(f"Validated Data: {serializer.validated_data}")  # 검증된 데이터를 출력하여 확인
            updateCompanyReport = self.companyReportService.updateCompanyReport(pk, serializer.validated_data)
            return Response(CompanyReportSerializer(updateCompanyReport).data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

    def readCompanyReportFinance(self, request):
        companyName = request.data.get('companyReportName')
        companyReportFinance = self.companyReportService.readCompanyReportFinance(companyName)

        # companyReportFinance는 (리스트2021, 리스트2022, 리스트2023) 형태이다.
        # 이를 하나의 리스트로 합쳐서 반환한다.
        combinedFinanceData = {
            '2021': companyReportFinance[0],
            '2022': companyReportFinance[1],
            '2023': companyReportFinance[2],
        }

        return Response(combinedFinanceData)    # JsonResponse가 자동으로 처리해준다.
    

    def readCompanyReportInfo(self, request):
        companyName = request.data.get('companyReportName')
        companyReportInfo = self.companyReportService.readCompanyReportFinance(companyName)
        return Response(companyReportInfo)  # JsonResponse가 자동으로 처리해준다.
    

    def readTopClickedCompany(self, request):
        topN = request.data.get('params').get('topN')
        topNCompanyId = self.companyReportService.readTopNCompany(topN)

        return Response(topNCompanyId)
    

    def updateReport(self, request):
        if request.data.get('aiResult'):
            data = request.data['aiResult']
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, "../../assets/report.json")
            with open(json_path, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
        
        try:
            self.companyReportService.updateCompanyReportDB(data)
            return Response(status=status.HTTP_200_OK)
        except FileNotFoundError:
            return Response({"error": "report.json 파일을 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)
    

    def saveKeyword(self, request):
        self.companyReportService.saveKeyword()
        return Response(status=status.HTTP_200_OK)