FROM ubuntu:xenial

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Update Ubuntu Software repository
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get upgrade -y
RUN apt-get install -y openssh-server vim locales zip
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev git
RUN locale-gen en_US.UTF-8
RUN update-locale

COPY python_install.sh /tmp
COPY build_partseg.sh /tmp
COPY create_build.py /tmp

RUN bash /tmp/python_install.sh


ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8