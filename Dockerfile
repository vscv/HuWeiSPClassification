# TF run on Docker NCHC 2018-11-05 
FROM ubuntu:16.04
RUN apt-get -y update && apt-get install -y python3-tk
COPY TF_run /home/TF_run
WORKDIR /home/TF_run
CMD ./Check.exe /home/TF_io/
RUN echo "tf_docker.v2 2018-11-05"
RUN echo "***** DL@NCHC *****"
