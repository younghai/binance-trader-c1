FROM continuumio/miniconda3:latest

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install build-essential -y \
  && apt-get install -y \
    git-core \
    vim \
    gettext-base \
    postgresql-client \
    python-psycopg2 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN conda install --yes pytorch torchvision cpuonly -c pytorch && conda clean -afy
RUN rm -rf /app/*

ENV PYTHONPATH $PYTHONPATH:/app:/app/src:/app/dev/src: