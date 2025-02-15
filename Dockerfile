FROM python:3.10

RUN apt update && apt install -y git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
