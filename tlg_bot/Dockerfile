FROM debian:stable-slim

RUN apt-get update

RUN apt-get install -y build-essential

RUN apt-get install -y python3 python3-pip python3-dev python3-venv ffmpeg

# Audio
# RUN apt-get install -y ffmpeg libopus-dev

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

ADD ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir -p /code /config /db /log /temp /assets

ADD . /code

WORKDIR /code

CMD ["bash"]
