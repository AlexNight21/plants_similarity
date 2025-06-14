FROM python:3.10.16-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update

WORKDIR /plants_similarity

COPY . .
RUN pip install --no-cache-dir -r requirements.txt