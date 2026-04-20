# CLVSA

We aim to reproduce the core findings of [CLVSA: A Convolutional LSTM-Based Variational Sequence-to-Sequence Model with Attention for Predicting Trends of Financial Markets (Wang et al., 2019)](https://arxiv.org/abs/2104.04041).

This repository provides an open-source PyTorch implementation of our interpretation of the model.

## Getting Started

Install packages with
`pip install -r requirements.txt`
Note that you may have to change the PyTorch installation if you want to use CUDA.

Set your `HF_TOKEN` in a `.env` file in the root directory to speed up the dataset download, which could take a few minutes. Note that it will be cached locally after the first download.

To recieve a summary of the dataset, run `python dataset/load_dataset.py`.

To train the model, run `python train.py`. You can run it with optional flags to tweak training parameters. See the internals of `train.py` for details.

## References

Jia Wang, Tong Sun, Benyuan Liu, Yu Cao, and Hongwei Zhu. *CLVSA: A Convolutional LSTM-Based Variational Sequence-to-Sequence Model with Attention for Predicting Trends of Financial Markets*. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 3705–3711, 2019.