#!/bin/bash
dir_path="$(dirname ${BASH_SOURCE[0]})"

if [ -z $1 ]; then env_name="buzzwords"; else env_name=$1; fi

conda create -y -n $env_name \
    -c rapidsai -c nvidia -c conda-forge \
    rapids=21.10 python=3.7 cudatoolkit=11.0 faiss-gpu==1.7.0;

source activate $env_name;

pip3 install \
    hdbscan==0.8.27 \
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