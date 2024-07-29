FROM python:3.9

WORKDIR /opt

RUN apt-get update && apt-get install -y sudo

RUN pip install --upgrade pip

COPY requirements.txt /opt/
RUN pip install -r requirements.txt

WORKDIR /work

COPY slanalytics.py /work/
COPY appfile /work/appfile

CMD ["streamlit", "run", "slanalytics.py", "--server.port", "8888"]