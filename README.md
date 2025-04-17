# MNIST Accuracy-per-Parameter Record

This repository contains a PyTorch implementation of a compact convolutional neural network (CNN) that achieves a **new state-of-the-art accuracy-per-parameter score on the MNIST dataset**.
It has 2 versions, both break the record (atleast after my current research about the record which I have done).

## mnis_small.py

Epoch 25/80 - Loss: 0.8046 - Accuracy: 73.63%
Test Loss: 0.7910 - Test Accuracy: 74.14%

His parameter-count is 232 and achieved around 74.14% accuracy. Which results around 0.0032 (0.32%) accuracy-per-parameter.

## mnis_normal.py

Epoch 30/80 - Loss: 0.5123 - Accuracy: 85.65%
Test Loss: 0.4862 - Test Accuracy: 86.05%

His parameter-count is 552 and achieved around 86.05% accuracy. Which results around 0.0016 (0.16%) accuracy-per-parameter.

## Notes

It is really weird that it worked that well, I was just testing and playing around with this because of a different problem on which I was testing on, but then I found this architecture (which is just a very simple minimal CNN).

```bash
pip install -r requirements.txt
