FROM ubuntu

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install build-essential -y && \
    apt-get install -y git && \
    apt-get install -y python-is-python3 python3.pip && \
    apt-get install -y ffmpeg && \
    apt-get install -y libsndfile1 && \
    apt-get install -y libsndfile1-dev

RUN pip install --upgrade pip
RUN pip cache purge

RUN pip install git+https://github.com/openai/whisper.git && \
    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git && \
    pip install setuptools==59.5.0

RUN pip install git+https://github.com/pyannote/pyannote-audio.git

COPY . ./pysper
WORKDIR ./pysper
RUN pip install -r ./requirement.txt

ENTRYPOINT ["python", "/pysper/pysper.py", "shell=True"]
