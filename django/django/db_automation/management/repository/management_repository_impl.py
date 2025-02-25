from django.db.models import Count
from account.entity.account import Account
from account.entity.account_role_type import AccountRoleType
from account.entity.login_history import LoginHistory
from account.repository.profile_repository_impl import ProfileRepositoryImpl
from management.repository.management_repository import ManagementRepository
from account.entity.profile import Profile
from marketing.entity.marketing import Marketing
from orders.entity.orders import Orders


from datetime import datetime, date, timedelta




class ManagementRepositoryImpl(ManagementRepository):
    __instance = None


    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__profileRepository = ProfileRepositoryImpl.getInstance()
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        
        return cls.__instance
    

    def userList(self):
        # Profile과 Account 정보를 가져오기 위한 쿼리
        profiles = Profile.objects.select_related('account', 'gender').all()

        # 결과를 프런트엔드가 원하는 형식으로 변환
        userList = []
        for profile in profiles:
            user_data = {
                'userId': profile.account.id,   # Account의 id
                'email': profile.email,
                'gender': profile.gender.gender_type,   # ProfileGenderType에서 gender 이름 가져오기
                'birthyear': profile.birthyear,
                'login_type': profile.account.loginType.loginType,  # AccountLoginType에서 로그인 타입 이름 가져오기
                'role_type': profile.account.roleType.roleType, # AccountRoleType에서 role 타입 이름 가져오기
                'last_login': profile.last_login,
            }
            userList.append(user_data)
        
        print(userList)
        return userList
    

    def grantRoleType(self, email):
        profile = self.__profileRepository.findByEmail(email)
        admin_role = AccountRoleType.objects.get(roleType='ADMIN')
        profile.account.roleType = admin_role
        profile.account.save()

        return profile
    

    def revokeRoleType(self, email):
        profile = self.__profileRepository.findByEmain(email)
        NORMAL = AccountRoleType.objects.get(roleType='NORMAL')
        profile.account.roleType = NORMAL
        profile.account.save()

        return profile
    

    def userLogList(self):
        # 모든 마케팅 데이터를 조회
        userLogList = Marketing.objects.all()

        # 조회한 데이터를 리스트에 담아 반환할 데이터 가공
        data = []
        for userLog in userLogList:
            user_data = {
                'userId': userLog.account.id,   # Account의 id
                'companyReportId': userLog.product.pk,  # CompanyReport의 id
                'companyReportName': userLog.product.companyReportName,
                'clickCount': userLog.click_count,  # 클릭 수
                'purchases': userLog.purchase,  # 구매 여부
                'lastClickDate': userLog.last_click_date.strftime('%Y-%m-%d %H:%M:%S'), # 마지막 클릭 시간
            }
            data.append(user_data)
        
        return data
    

    def userLogData(self):
        account = Account.objects.all()
        today = date.today()  # 오늘의 날짜
        start_date = datetime.combine(today - timedelta(days=7), datetime.min.time())  # 시작 시간을 오늘의 자정으로 설정
        end_date = datetime.combine(today + timedelta(days=7), datetime.min.time())  # 종료 시간을 오늘에서 1주일 후의 자정으로 설정

        # 로그인 기록에서 계정 수 필터링
        filtered_history = LoginHistory.objects.filter(login_at__range=(start_date, end_date)).values(
            'account_id').distinct().count()

        # 주문 수 필터링
        orders = Orders.objects.filter(createdDate__range=(start_date, end_date)).values(
            'account_id').distinct().count()

        # 두 번 이상 주문한 계정 수 필터링
        two_or_more_orders = Orders.objects.filter(
            createdDate__range=(start_date, end_date)
        ).values('account_id').annotate(order_count=Count('id')).filter(order_count__gte=2).count()

        # 각 통계 값 계산
        userCount = len(account)
        accountCount = filtered_history
        purchaseCount = orders
        revenueCount = two_or_more_orders

        # userData를 딕셔너리 형태로 구성
        userData = {
            'userCount': int(userCount),  # 사용자 수
            'accountCount': int(accountCount),  # 유일한 계정 수
            'purchaseCount': int(purchaseCount),  # 총 주문 수
            'revenueCount': int(revenueCount)  # 두 번 이상 구매한 계정 수
        }

        return userData  # userData 딕셔너리 반환