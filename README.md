# BIMPM-pytorch
Re-implementation of [BIMPM](https://arxiv.org/abs/1702.03814) for [SNLI](https://nlp.stanford.edu/projects/snli/) on Pytorch

## Results

(Dataset: SNLI)

| Model        |  ACC(%)   | 
|--------------|:----------:|
| **Re-implementation (with default parameters)** | **86.3** |  
| Baseline from the paper (Single BiMPM)          |  86.9    |    


## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2.
- Pytorch: 0.3.0

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    nltk==3.2.4
    torchtext==0.2.0
    torch==0.3.0
    tensorboardX==0.8

## Training

> python train.py --help

	usage: train.py [-h] [--batch-size BATCH_SIZE] [--char-dim CHAR_DIM]
                [--char-hidden-size CHAR_HIDDEN_SIZE] [--dropout DROPOUT]
                [--epoch EPOCH] [--gpu GPU] [--hidden-size HIDDEN_SIZE]
                [--learning-rate LEARNING_RATE]
                [--num-perspective NUM_PERSPECTIVE] [--use-char-emb]
                [--word-dim WORD_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --char-dim CHAR_DIM
      --char-hidden-size CHAR_HIDDEN_SIZE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --learning-rate LEARNING_RATE
      --num-perspective NUM_PERSPECTIVE
      --use-char-emb
      --word-dim WORD_DIM

## Test

> python test.py --help

	usage: train.py [-h] [--batch-size BATCH_SIZE] [--char-dim CHAR_DIM]
                [--char-hidden-size CHAR_HIDDEN_SIZE] [--dropout DROPOUT]
                [--epoch EPOCH] [--gpu GPU] [--hidden-size HIDDEN_SIZE]
                [--learning-rate LEARNING_RATE]
                [--num-perspective NUM_PERSPECTIVE] [--use-char-emb]
                [--word-dim WORD_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --char-dim CHAR_DIM
      --char-hidden-size CHAR_HIDDEN_SIZE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --learning-rate LEARNING_RATE
      --num-perspective NUM_PERSPECTIVE
      --use-char-emb
      --word-dim WORD_DIM
      --model-path MODEL_PATH

	
Note: You should execute **test.py** with the same hyperparameters, which are used for training the model you want to run.    
