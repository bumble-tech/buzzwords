#!/bin/bash
dir_path="$(dirname ${BASH_SOURCE[0]})"

if [ -z $1 ]; then env_name="buzzwords"; else env_name=$1; fi

conda create -y -n $env_name \
    -c rapidsai -c nvidia -c conda-forge \
    cuml=22.10 python=3.8 cudatoolkit=11.5;

source activate $env_name;

pip3 install \
    sentence-transformers==2.1.0 \
    keybert==0.5.1 \
    pytest~=7.0.0 \
    clip-by-openai==1.1;

python3 -m nltk.downloader \
    punkt \
    wordnet \
    averaged_perceptron_tagger \
    stopwords \
    "omw-1.4";

pip3 install -e $dir_path