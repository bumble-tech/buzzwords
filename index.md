# Home

Buzzwords is Bumble's open-source GPU-powered topic modelling tool, developed inhouse and building upon the work found in [BERTopic](https://maartengr.github.io/BERTopic/index.html) and [Top2Vec](https://arxiv.org/abs/2008.09470). The underlying algorithm is effectively the same, but we streamlined the process and used recent developments in GPU-powered libraries to speed up the model significantly. We found that the current implementations were not fast enough for our purposes, and it was unfeasible to use out-of-the-box solutions for the scale that we need.

Check out our [article](https://medium.com/bumble-tech/multilingual-gpu-powered-topic-modelling-at-scale-dc8bd08609ef) for some more background about the project!

Buzzwords is by no means a finished project and can be improved massively. If you run into any issues while using Buzzwords, please do reach out to use or raise an issue on Github - your help is always appreciated

* [Documentation](documentation/)
* [General Info](info/)
* [Installation](installation/)
* [Tutorial - Basic Usage](tutorial/)
