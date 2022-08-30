# Buzzwords

Buzzwords is Bumble's GPU-powered topic modelling tool, used for gathering insights on topics in text or images on a large scale. The algorithm is based on [Bertopic](https://maartengr.github.io/BERTopic/index.html) and [Top2Vec](https://arxiv.org/abs/2008.09470), but altered to be faster.

For more information see [the website](https://bumble-tech.github.io/buzzwords/)

## Installation

Installation for buzzwords is somewhat complicated, due to the need for RAPIDS.ai (and to a lesser extent, FAISS) on an Nvidia GPU-powered machine. RAPIDS doesn't support installation through pip anymore, so we need to use conda environments.

For ease of installation, we've packaged it up into a bash script `install.sh`

```bash
$ ./install.sh buzzwords
```

This will create the conda environment (with either your given name or `buzzwords` as default) with Buzzwords installed in it

## Basic Examples

To instantiate the model is very simple

```python
from buzzwords import Buzzwords
 
model = Buzzwords()
```

To train the model on a set of documents, call the fit_transform() function to return the topics

```python
docs = df['text_column']
 
topics = model.fit_transform(docs)
```
