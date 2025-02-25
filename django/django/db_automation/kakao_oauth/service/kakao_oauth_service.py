from abc import ABC, abstractmethod



class KakaoOauthService(ABC):

    @abstractmethod
    def kakaoLoginAddress(self):
        pass


    @abstractmethod
    def requestAccessToken(self, kakaoAuthcode):
        pass


    @abstractmethod
    def requestUserInfo(self, accessToken):
        pass