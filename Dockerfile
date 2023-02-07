FROM python:3.8

RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 5000
CMD ["python", "/code/app.py"]