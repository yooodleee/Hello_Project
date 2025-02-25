import redis
from django.conf import settings


from redis_cache.service.redis_cache_service import RedisCacheService



class RedisCacheServiceImpl(RedisCacheService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__init__(cls)
            cls.__instance.redisClient = redis.StrictRedis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
        
        return cls.__instance
    

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    

    def storeKeyValue(self, key, value):
        try:
            self.redisClient.set(key, value)
        except Exception as e:
            print('Error storing access token in Redis: ', e)
            raise e
    

    def getValueByKey(self, key):
        try:
            return self.redisClient.get(key)
        except Exception as e:
            print("redis key로 value 찾는 중 에러 발생: ", e)
            raise e
    

    def deleteKey(self, key):
        try:
            result = self.redisClient.delete(key)
            if result == 1:
                print(f"유저 토큰 삭제 성공: {key}")
                return True
            
            return False
        except Exception as e:
            print("redis key 삭제 중 에러 발생: ", e)
            raise e