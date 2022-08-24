FROM gcr.io/deeplearning-platform-release/base-cu110

COPY buzzwords/ buzzwords/

COPY install.sh setup.py ./

RUN chmod +x ./install.sh

RUN bash ./install.sh

CMD ["/bin/bash"]