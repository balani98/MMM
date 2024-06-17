FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True

ENV APP_HOME=/app

ENV PORT=8001

WORKDIR $APP_HOME

COPY . ./
RUN apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler



# Install Redis server
RUN apt-get update && \
    apt-get install -y redis-server


# Install R and necessary system libraries
RUN apt-get update && apt-get install -y \
    r-base \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    cmake 

# Create a group and user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Change ownership of the directory to the appuser
RUN chown -R appuser:appgroup /app



# set the env var
ENV ENVIRONMENT=DOCKER

RUN Rscript -e "install.packages('reticulate')" \
    -e "library(reticulate)" \
    -e "virtualenv_create('r-reticulate')" \
    -e "use_virtualenv('r-reticulate', required = TRUE)" \
    -e "Sys.setenv(RETICULATE_PYTHON = '/usr/local/bin/python3.10')" \
    -e "py_install('nevergrad', pip = TRUE)" \
    -e "py_install('numpy', pip = TRUE)"

RUN Rscript -e "install.packages(c('base','curl','xml2','utils','httr','Matrix','openssl','openxlsx','nolptr','glmnet','lares','lubridate','timeDate','deplr','ggplot2','remotes','Robyn','doParallel'))"
#RUN Rscript -e "install_github('facebookexperimental/Robyn/R')"

# You can add more R package installations here if needed

# Expose the Flask port
EXPOSE 8001

# Install any needed packages specified in requirements.txt for Python
RUN pip install --no-cache-dir -r requirements.txt

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
#CMD ["bash", "-c", "service redis-server start && celery -A tasks worker --loglevel=info & python main.py"]# Start Redis server and Celery worker, then run the main application
CMD ["bash", "-c", "service redis-server start && celery -A tasks worker --loglevel=info & gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app"]

# Add health check to monitor the container's status
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost/health || exit 1