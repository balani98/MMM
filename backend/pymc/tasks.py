from pymc_starter import build_pymc_model
from celery import Celery
from celery.result import AsyncResult
import celery.states as states
import redis
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
celery_app.conf.update(
    broker_connection_retry_on_startup=True
)
# long running task of predictor model
@celery_app.task()
def build_model_output_with_celery(body, filename):
    clear_redis_cache()
    build_pymc_model(body,filename)
    return "Task completed successfully and model has been built!"

@celery_app.task()
def clear_redis_cache():
    # Connect to Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # Flush all keys from the Redis cache
    redis_client.flushall()

    print("Redis cache cleared successfully!")