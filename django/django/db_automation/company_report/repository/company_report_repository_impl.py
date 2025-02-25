import json
import os
import re
import random
from django.db.models import Sum


from company_report.entity.company_data_finance import FinancialData
from company_report.entity.company_data_total import CompanyDataTotal
from company_report.entity.company_report import CompanyReport
from company_report.repository.company_report_repository import CompanyReportRepository
from db_automation.db_automation import settings
from marketing.entity.marketing import Marketing



class CompanyReportRepositoryImpl(CompanyReportRepository):
    __instance = None
    LABEL_KEYWORDS = {
        '플랫폼': {
            'keywords': ['플랫폼'],
            'exclude': []
        },
        '빅데이터': {
            'keywords': ['빅데이터', '머신러닝', '딥러닝'],
            'exclude': []
        },
        '정보보안': {
            'keywords': ['정보보안'],
            'exclude': []
        },
        '소프트웨어': {
            'keywords': ['소프트웨어'],
            'exclude': ['반도체']
        },
        '하드웨어': {
            'keywords': ['제조 서비스', '하드웨어'],
            'exclude': ['자동차', 'AI']
        },
        '클라우드': {
            'keywords': ['클라우드'],
            'exclude': ['슈퍼마켓']
        },
        '제조': {
            'keywords': ['제조'],
            'exclude': ['백화점', '오뚜기', '개인정보']
        },
        '컨설팅': {
            'keywords': ['솔루션 제공'],
            'exclude': []
        },
        '헬스케어': {
            'keywords': ['헬스케어'],
            'exclude': ['카카오']
        },
        '게임': {
            'keywords': ['게임'],
            'exclude': ['웹툰', '신세계']
        },
        '메타버스': {
            'keywords': ['메타버스'],
            'exclude': []
        },
        '인프라': {
            'keywords': ['IT 인프라', '정보시스템'],
            'exclude': []
        },
        '반도체': {
            'keywords': ['반도체'],
            'exclude': []
        },
        '화학': {
            'keywords': ['화학'],
            'exclude': []
        },
        '의료': {
            'keywords': ['의약품'],
            'exclude': []
        },
        'AI': {
            'keywords': ['인공지능'],
            'exclude': ['광고', '세일즈']
        },
        '자동차':{
            'keywords': ['자동차'],
            'exclude': ['']
        },
        '디스플레이': {
            'keywords': ['디스플레이'],
            'exclude': ['광고']
        },
        '마케팅/광고': {
            'keywords': ['광고'],
            'exclude': []
        },
        '영상 분석': {
            'keywords': ['영상'],
            'exclude': ['광고']
        },
        '네트워크': {
            'keywords': ['네트워크'],
            'exclude': ['전자부품', '항체의약품', '부품 제조', '방사선', '핵융합']
        },
        '식품': {
            'keywords': ['식품'],
            'exclude': ['건강기능식품']
        },
        '쇼핑': {
            'keywords': ['쇼핑'],
            'exclude': ['IT']
        },
        '배터리': {
            'keywords': ['배터리'],
            'exclude': []
        },
        '건설': {
            'keywords': ['건설'],
            'exclude': ['마트']
        },
        '호텔': {
            'keywords': ['호텔'],
            'exclude': []
        },
        '금융지원': {
            'keywords': ['금융'],
            'exclude': ['반도체', '음반', '금융권', '건강기능식품']
        }
    }


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance


    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance


    def list(self):
        return CompanyReport.objects.all().order_by('companyReportName')


    def create(self, companyReportName, companyReportPrice, companyReportCategory, content, companyReportTitleImage):
        uploadDirectory='../../AIM-Sniper-frontend/src/assets/images/uploadimages'
        print('업로드된 디렉토리 : ', uploadDirectory)
        os.makedirs(uploadDirectory, exist_ok=True)

        imagePath = os.path.join(uploadDirectory, companyReportTitleImage.name)
        with open(imagePath, 'wb+') as destination:
            for chunk in companyReportTitleImage.chunks():
                destination.write(chunk)
        print('이미지 경로: ',imagePath)


        companyReport = CompanyReport(
            companyReportName=companyReportName,
            content=content,
            companyReportPrice=companyReportPrice,
            companyReportCategory=companyReportCategory,
            companyReportTitleImage=companyReportTitleImage.name
        )
        companyReport.save()
        return companyReport


    def findByCompanyReportId(self, companyReportId):
        try:
            return CompanyReport.objects.get(companyReportId=companyReportId)
        except CompanyReport.DoesNotExist:
            return None


    def findByCompanyReportIdList(self, companyReportIdList):
        try:
            return CompanyReport.objects.filter(companyReportId__in=companyReportIdList)
        except CompanyReport.DoesNotExist:
            return None


    def findAllByCompanyReportCategory(self, companyReportCategory):
        try:
            return CompanyReport.objects.filter(companyReportCategory=companyReportCategory)
        except:
            return None


    def deleteByCompanyReportId(self, companyReportId):
        companyReport = CompanyReport.objects.get(companyReportId=companyReportId)
        companyReport.delete()


    def update(self, companyReport, companyReportData):
        for key, value in companyReportData.items():
            setattr(companyReport, key, value)
        companyReport.save()
        return companyReport


    def readCompanyReportFinance(self, companyReportName):
        # 해당 회사와 연도의 데이터를 필터링
        id=CompanyDataTotal.objects.get(company_name=companyReportName)
        FinanceData2021 = FinancialData.objects.filter(company=id, year=2021).values('revenue', 'profit_trend','owners_capital')
        FinanceData2022 = FinancialData.objects.filter(company=id, year=2022).values('revenue', 'profit_trend', 'owners_capital')
        FinanceData2023 = FinancialData.objects.filter(company=id, year=2023).values('revenue', 'profit_trend', 'owners_capital')
        # 결과 반환
        return list(FinanceData2021),list(FinanceData2022),list(FinanceData2023)


    def readCompanyReportInfo(self, companyReportName):
        companyReportInfo = CompanyDataTotal.objects.filter(company_name=companyReportName)
        basicInfo=companyReportInfo.values(
            'company_name','est_date','company_class','ceo_name','address','website','business_summary','revenue_table')
        return basicInfo


    def readTopNCompany(self, topN):
        # companyReport = CompanyReport.objects.all()
        sortedCompany = Marketing.objects.values('product').annotate(Sum('click_count')).order_by('-click_count__sum')
        clickedTopNCompany = [dict['product']
                                for dict in sortedCompany[:topN].values('product')]

        return clickedTopNCompany


    def saveDataToCompanyTotalDB(self, corpName, corpData):
        company, created = CompanyDataTotal.objects.get_or_create(
            company_name=corpName,
            defaults={
                "est_date": corpData["est_dt"],
                "company_class": corpData["corp_cls"],
                "ceo_name": corpData["ceo_nm"],
                "address": corpData["adres"],
                "website": corpData["hm_url"],
                "business_summary": corpData["businessSummary"],
                "revenue_table": corpData["revenueTable"],
            }
        )

        if not created:
            company.est_date = corpData["est_dt"]
            company.company_class = corpData["corp_cls"]
            company.ceo_name = corpData["ceo_nm"]
            company.address = corpData["adres"]
            company.website = corpData["hm_url"]
            company.business_summary = corpData["businessSummary"]
            company.revenue_table = corpData["revenueTable"]

            company.save()


    def getDataFromFinanceKeys(self, financeDict, index):
        return list(financeDict.keys())[index]


    def getDataFromFinanceValues(self, financeDict, index):
        return list(financeDict.values())[index]


    def saveDataToCompanyFinanceDB(self, corpName, corpData):
        try:
            company = CompanyDataTotal.objects.get(company_name=corpName)

            loopMax = len(corpData['revenueTrend'])
            for index in range(loopMax):
                finance, created = FinancialData.objects.update_or_create(
                    company=company,
                    year=self.getDataFromFinanceKeys(corpData['revenueTrend'], index),
                    defaults={
                        "revenue": self.getDataFromFinanceValues(corpData['revenueTrend'], index),
                        "profit_trend": self.getDataFromFinanceValues(corpData['profitTrend'], index),
                        "owners_capital": self.getDataFromFinanceValues(corpData['ownersCapital'], index),
                    }
                )
                company.save()

        except CompanyDataTotal.DoesNotExist:
            print(f"--> Error: 회사 '{corpName}'가 존재하지 않습니다. 먼저 회사를 저장하세요.")


    def saveDataToCompanyReport(self, corpName):
        try:
            companyName = corpName
            productPrice = 5

            # 파일이 존재하는지 확인 후 productImage에 값 할당
            productImage = f"{companyName}.png"

            # business_summary 필드 값을 가져옴
            content = CompanyDataTotal.objects.filter(company_name=companyName).values('business_summary').first()
            id = CompanyDataTotal.objects.get(company_name=companyName)

            # 매출 데이터 가져오기
            finance_data_list = []
            for year in [2021, 2022, 2023]:
                finance_data = FinancialData.objects.filter(company=id, year=year).values('revenue').first()

                # 데이터 타입이 딕셔너리인지 확인
                if isinstance(finance_data, dict):
                    revenue = finance_data.get('revenue')
                    if revenue != 0:  # 매출액이 0이 아닌 경우만 추가
                        finance_data_list.append(revenue)
                elif isinstance(finance_data, int) and finance_data != 0:
                    # finance_data가 int 타입일 경우 값이 0이 아니면 추가
                    finance_data_list.append(finance_data)

            # 평균 매출 계산 (데이터가 있을 때만 계산)
            if finance_data_list:
                meanFinanceData = sum(finance_data_list) / len(finance_data_list)

                # 회사 규모 분류
                if meanFinanceData >= 1_000_000_000_000:  # 예: 연 매출 1조 이상이면 대기업
                    companyCategory = '매출액 1조 이상'
                elif meanFinanceData >= 100_000_000_000:  # 예: 연 매출 5000억 이상이면 중견기업
                    companyCategory = '매출액 1000억 이상 1조 미만'
                else:  # 예: 연 매출 5000억 미만이면 중소기업
                    companyCategory = '매출액 1000억 미만'
            else:
                companyCategory = '그 외'  # 매출 데이터가 전혀 없을 경우

            # CompanyReport 객체가 존재하면 업데이트, 없으면 생성
            CompanyReport.objects.update_or_create(
                companyReportName=companyName,  # 조건에 맞는 데이터를 찾을 필드
                defaults={
                    'companyReportPrice': productPrice,
                    'companyReportTitleImage': productImage,
                    'companyReportCategory': companyCategory,
                    'content': content
                }
            )

        except CompanyDataTotal.DoesNotExist:
            print(f"--> Error: 회사 '{corpName}'가 존재하지 않습니다. 먼저 회사를 저장하세요.")


    def updateDataToDB(self, data):
        companyData = data

        if companyData == None:
            with open("./assets/report.json", "r", encoding="utf-8-sig") as file:
                companyData = json.load(file)

        for corpName in companyData.keys():
            try:
                self.saveDataToCompanyTotalDB(corpName, companyData[corpName])
            except Exception as e:
                print(f"* Total Save Fail ({corpName}) -> {e}")

            try:
                self.saveDataToCompanyFinanceDB(corpName, companyData[corpName])
            except Exception as e:
                print(f"* Finance Save Fail ({corpName}) -> {e}")
            try:
                self.saveDataToCompanyReport(corpName)
                self.label_and_save_keyword()
            except Exception as e:
                print(f"* Company Report Fail ({corpName}) -> {e}")


    def label_and_save_keyword(self):
        """주어진 데이터 리스트에 대해 intent 라벨링을 하고, CompanyReport 엔티티의 keyword에 저장"""
        # CompanyDataTotal의 모든 데이터를 가져와 특수 문자를 제거한 후 summaries 리스트 생성
        summaries = [{'companyName': item['company_name'],
                      'businessSummary': re.sub(r'(\*\*|\*|-|\\n|\n)', '', str(item['business_summary'])).strip()}
                     for item in CompanyDataTotal.objects.all().values('company_name', 'business_summary')]

        labeled_count, unlabeled_count = 0, 0

        # 각 summary에 대해 처리
        for summary in summaries:
            # 라벨링 규칙에 따른 라벨 추출
            labels = [
                label for label, rules in self.LABEL_KEYWORDS.items()
                if any(keyword in summary['businessSummary'] for keyword in rules['keywords'])
                   and not any(exclusion in summary['businessSummary'] for exclusion in rules['exclude'])
            ]

            # 라벨이 없으면 "기타"로 설정
            if not labels:
                labels = ["기타"]

            summary['rule_based_intent'] = labels  # 라벨링 결과를 summary에 저장

            # CompanyReport 엔티티에서 해당 회사 이름에 맞는 레포트를 가져옴
            company_report = CompanyReport.objects.filter(companyReportName=summary['companyName']).first()

            # CompanyReport의 keyword가 비어 있거나 기존 keyword가 없는 경우에만 저장
            if company_report:
                if not company_report.keyword:  # keyword가 없으면 라벨을 저장
                    company_report.keyword = ','.join(labels)  # 리스트를 문자열로 변환하여 저장
                    company_report.save()
                elif "기타" in labels:  # 이미 keyword가 있더라도 "기타"는 덧붙이지 않도록 설정 가능
                    unlabeled_count += 1
                else:
                    labeled_count += 1

        return summaries, labeled_count, unlabeled_count