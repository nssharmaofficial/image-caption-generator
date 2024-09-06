import os
import torch


class Config(object):

    def __init__(self) -> None:

        self.DEVICE = torch.device("cuda:0")  # Device to run the model on

        self.BATCH = 32                       # Batch size for training
        self.EPOCHS = 5                       # Number of training epochs

        self.VOCAB_FILE = 'vocabulary.txt'   # File containing vocabulary mappings
        self.VOCAB_SIZE = 5000                # Size of the vocabulary

        self.NUM_LAYER = 1                    # Number of LSTM layers
        self.IMAGE_EMB_DIM = 512              # Dimension of image features
        self.WORD_EMB_DIM = 512               # Dimension of word embeddings
        self.HIDDEN_DIM = 512                 # Dimension of LSTM hidden states
        self.LR = 0.001                       # Learning rate for training

        # Weights used to predict sample
        self.EMBEDDING_WEIGHT_FILE = 'code/checkpoints/embeddings-32B-512H-1L-e5.pt'
        self.ENCODER_WEIGHT_FILE = 'code/checkpoints/encoder-32B-512H-1L-e5.pt'
        self.DECODER_WEIGHT_FILE = 'code/checkpoints/decoder-32B-512H-1L-e5.pt'

        self.ROOT = os.path.join(os.path.expanduser('~'), 'Github', 'ImageCaption_Flickr30k')
