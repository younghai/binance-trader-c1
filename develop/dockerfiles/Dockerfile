FROM pytorch/pytorch:latest

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install build-essential -y \
  && apt-get install -y \
    git-core \
    vim \
    gettext-base \
    ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
ENV PYTHONPATH $PYTHONPATH:/app:/app/src:/develop:/develop/src
