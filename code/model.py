import torch
import torch._utils
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from config import *


class Encoder(nn.Module):

    def __init__(self, image_emb_dim:int, device:torch.device):
        """ Image encoder to obtain features from images. Contains pretrained Resnet50 with last layer removed 
            and a linear layer with the output dimension of (BATCH, image_emb_dim)

        Args:
            image_emb_dim (int): final output dimension of features
            
            device (torch.device)
        """
        
        super(Encoder, self).__init__()
        self.image_emb_dim = image_emb_dim
        self.device = device
        
        print(f"Encoder:\n \
                Encoder dimension: {self.image_emb_dim}")
        
        # pretrained Resnet50 model with freezed parameters
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters(): 
            param.requires_grad_(False)
        
        # remove last layer 
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # define a final classifier
        self.fc = nn.Linear(resnet.fc.in_features, self.image_emb_dim) 
        
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward operation of encoder, passing images through resnet and then linear layer.

        Args:
            > images (torch.Tensor): (BATCH, 3, 224, 224)

        Returns:
            > features (torch.Tensor): (BATCH, IMAGE_EMB_DIM)
        """
        
        features = self.resnet(images)
        # features: (BATCH, 2048, 1, 1)
        
        features = features.reshape(features.size(0), -1).to(self.device)
        # features: (BATCH, 2048)
        
        features = self.fc(features).to(self.device)
        # features: (BATCH, IMAGE_EMB_DIM)
        
        return features
    

class Decoder(nn.Module):
    
    def __init__(self, image_emb_dim:int, word_emb_dim:int, hidden_dim:int, num_layers:int, vocab_size:int, device:torch.device):
        """ Decoder taking as input for the LSTM layer the concatenation of features obtained from the encoder 
        and embedded captions obtained from the embedding layer. Hidden and cell states are zero initialized.
        Final classifier is a linear layer with output dimension of the size of a vocabulary.

        Args:
            image_emb_dim (int): the dimension of features obtained from the encoder
            
            word_emb_dim (int): the dimension of word embeddings from embedding layer
            
            hidden_dim (int): capacity of LSTM (dimension: image_emb_dim + word_emb_dim)
            
            num_layers (int): number of LSTM layers
            
            vocab_size (int): out_features of linear layer
            
            device (torch.device)
        """
        
        
        super(Decoder,self).__init__()
        
        self.config = Config()
        
        self.image_emd_dim = image_emb_dim
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layers
        self.vocab_size = vocab_size
        self.device = device
        
        self.hidden_state_0 = nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_dim)))
        self.cell_state_0 = nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_dim)))
              
        print(f"Decoder:\n \
                Encoder Size:  {self.image_emd_dim},\n \
                Embedding Size: {self.word_emb_dim},\n \
                LSTM Capacity: {self.hidden_dim},\n \
                Number of layers: {self.num_layer},\n \
                Vocabulary Size: {self.vocab_size},\n \
                ")
        
        self.lstm = nn.LSTM(self.image_emd_dim + self.word_emb_dim, self.hidden_dim, num_layers=self.num_layer, bidirectional = False)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.vocab_size),
            nn.LogSoftmax(dim=2)  
        )
        
    def forward(self, embedded_captions: torch.Tensor, features: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Forward operation of (word-by-word) decoder. The LSTM input (concatenation of embedded_captions and features) is passed through LSTM and then linear layer.
        
        Args:
        
            > embedded_captions(torch.Tensor): (SEQ_LENGTH = 1, BATCH, WORD_EMB_DIM)
            > features (torch.Tensor): (1, BATCH, IMAGE_EMB_DIM)
            > hidden (torch.Tensor): (NUM_LAYER, BATCH, HIDDEN_DIM)
            > cell (torch.Tensor): (NUM_LAYER, BATCH, HIDDEN_DIM)

        Returns:
        
            > output (torch.Tensor): (1, BATCH, VOCAB_SIZE)
            > (hidden, cell) (torch.Tensor, torch.Tensor): (NUM_LAYER, BATCH, HIDDEN_DIM), (NUM_LAYER, BATCH, HIDDEN_DIM)
        """
        
        lstm_input = torch.cat((embedded_captions, features), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output : (length = 1, BATCH, HIDDEN_DIM)
        # hidden : (NUM_LAYER, BATCH, HIDDEN_DIM)
        
        output = output.to(self.device)
        
        output = self.fc(output)
        # output : (length = 1, BATCH, VOCAB_SIZE)
        
        return output, (hidden, cell)


def get_acc(output, target):
    # output: (BATCH, VOCAB_SIZE)
    # output: (BATCH, --WORD--)
    
    probability = torch.exp(output)
    
    # get the maximum probability among dim=1 (VOCAB_SIZE) -> returns: value, index
    # get the index ([1])
    equality = (target == probability.max(dim=1)[1])
    return equality.float().mean()
    
if __name__ == '__main__':
    
    config = Config()

    encoder = Encoder(image_emb_dim = config.IMAGE_EMB_DIM, device = config.DEVICE)
    emb_layer = torch.nn.Embedding(num_embeddings = config.VOCAB_SIZE, 
                                   embedding_dim = config.WORD_EMB_DIM,
                                   padding_idx = 0)
    decoder = Decoder(image_emb_dim = config.IMAGE_EMB_DIM,
                   word_emb_dim = config.WORD_EMB_DIM,
                   hidden_dim = config.HIDDEN_DIM,
                   num_layers = config.NUM_LAYER,
                   vocab_size = config.VOCAB_SIZE,
                   device = config.DEVICE)
    
    encoder = encoder.to(config.DEVICE)
    emb_layer = emb_layer.to(config.DEVICE) 
    decoder = decoder.to(config.DEVICE)
    
    # create random tensor of images and captions 
    # generating captions will be word by word so suppose captions have length = 1 (second dimension)
    images = torch.randn((32, 3, 256, 256)).to(config.DEVICE)
    captions = torch.randint(low=1, high=100, size=(32, 1), dtype=torch.int).to(config.DEVICE) 
    
    # pass images through encoder
    features = encoder.forward(images = images)
    features = features.unsqueeze(0)
    print('Features size: ', features.size()) # (1, BATCH, IMAGE_EMD_DIM)
    
    # pass captions through embedding layer
    embedded_captions = emb_layer.forward(captions)
    embedded_captions = embedded_captions.permute(1,0,2)  
    print('Embedded captions size: ', embedded_captions.size()) # (1, BATCH, WORD_EMB_DIM)
    
    # initialize hidden and cell to be of size: (NUM_LAYER, BATCH, HIDDEN_DIM)
    # note: HIDDEN_DIM = IMAGE_EMB_DIM + WORD_EMB_DIM
    hidden = decoder.hidden_state_0.repeat(1, 32, 1).to(config.DEVICE)
    cell = decoder.cell_state_0.repeat(1, 32, 1).to(config.DEVICE)
    
    # pass embedded captions and features through decoder (they will be concatenated)
    output, (hidden_state, cell_state) = decoder.forward(
        embedded_captions=embedded_captions,
        features=features,
        hidden=hidden,
        cell=cell
        )

    print('Output size: ', output.size())       # (1, BATCH, VOCAB_SIZE)
    print('Hidden size: ', hidden_state.size()) # (NUM_LAYER, BATCH, HIDDEN_DIM)
    print('Cell size: ', cell_state.size())     # (NUM_LAYER, BATCH, HIDDEN_DIM)

    
    
    
        
    

