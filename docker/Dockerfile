FROM gcr.io/deeplearning-platform-release/base-cu110@sha256:9717d641903840e9fc9b697190b0048d36e957f54e5c18eeb67a3fdfebd6c5b8

COPY buzzwords/ buzzwords/

COPY install.sh setup.py ./

RUN chmod +x ./install.sh

RUN bash ./install.sh

CMD ["/bin/bash"]