ARG BASE_IMAGE
FROM $BASE_IMAGE
USER root

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install build-essential -y \
  && apt-get install -y \
    git-core \
    vim \
    gettext-base \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN rm -rf /app/*

ENV PYTHONPATH $PYTHONPATH:/app:/app/src: