FROM python:3

RUN pip install -U pip wheel

RUN pip install --use-feature=2020-resolver comet_ml keras tensorflow

ADD train.py /train.py

