FROM python:3.10

ENV PYTHONUNBUFFERED=True

ENV APP_HOME=/app

ENV PORT=8003

WORKDIR $APP_HOME

COPY . ./

# set the env var
ENV ENVIRONMENT=DOCKER

# Expose the Flask port
EXPOSE 8003

# Install Redis server
RUN apt-get update && \
    apt-get install -y redis-server

# Install any needed packages specified in requirements.txt for Python
RUN pip install --no-cache-dir -r requirements.txt

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
#CMD ["bash", "-c", "service redis-server start && celery -A tasks worker --loglevel=info & python main.py"]# Start Redis server and Celery worker, then run the main application
CMD ["bash", "-c", "service redis-server start && celery -A tasks worker --loglevel=info & gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app"]

# Add health check to monitor the container's status
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost/health || exit 1