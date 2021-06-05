FROM python:3.9.4-slim-buster

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y install libssl-dev \
    && apt-get clean \
    && pip install --upgrade pip \
    && pip install wheel

# ports
# for jupyter, needs to be passed with -p xxxx:8888 at run time
EXPOSE 8888

# Prepare environment
RUN mkdir /tp
WORKDIR /tp

# Install dependencies
#COPY apt.txt /tp/
#RUN cat apt.txt | xargs apt-get install -y
COPY requirements.txt /tp/
RUN pip install -r requirements.txt --no-cache-dir

# install jupyter lab -- optional
RUN pip install jupyterlab
ENV SHELL=/bin/bash
# last line needed otherwise bug when opening terminal from jupyter.

# Install and execute
#COPY . /mm3/

# Prepare environment
RUN mkdir /user
WORKDIR /user
COPY  docker/resources/.bashrc /root/

#ENTRYPOINT ["python"]
#CMD jupyter lab --no-browser --ip 0.0.0.0 --allow-root
CMD /bin/bash
