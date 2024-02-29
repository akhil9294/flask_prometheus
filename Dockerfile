# FROM python:3.10.2-slim-buster
# WORKDIR /app
# COPY requirements.txt .
# RUN pip3 install -r requirements.txt
# COPY server.py .
# EXPOSE 5000
# ENTRYPOINT ["python","server.py"]


# FROM python:3.7.3-alpine3.9 as prod

FROM python:3.9.8 as prod
RUN mkdir /app/
WORKDIR /app/

COPY requirements.txt .
COPY std_scaler.bin .
COPY model_classifier.pkl .
COPY server.py .

RUN pip install -r requirements.txt

ENV FLASK_APP=server.py
CMD flask run -h 0.0.0.0 -p 5000