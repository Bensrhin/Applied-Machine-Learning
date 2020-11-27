# Recurrent Neural Networks: Sequence tagging

We use a dataset from the CoNLL conferences that benchmark natural language processing systems and tasks:
[Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)

## Objectives

* Load pre-trained word vectors, and measure similarity using cosine similarity.
* Apply stacking embeddings: GloVe embeddings + Traning arbitrary initilized emebddings.
* Experiment different architectures, namely RNN and LSTM.

### Installing

The results of all models suggest that GPU is better than CPU when the number
of parameters is high.
So make sure that you install the GPU version of TensorFlow backend:

```
$ pip install tensorflow-gpu
```

## Author

* **BENSRHIER Nabil**
