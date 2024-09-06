import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
from config import Config


class Encoder(nn.Module):
    """
    Image encoder to obtain features from images using a pretrained ResNet-50 model.
    The last layer of ResNet-50 is removed, and a linear layer is added to transform
    the output to the desired feature dimension.

    Args:
        image_emb_dim (int): Final output dimension of image features.
        device (torch.device): Device to run the model on (CPU or GPU).
    """

    def __init__(self, image_emb_dim: int, device: torch.device):
        super(Encoder, self).__init__()
        self.image_emb_dim = image_emb_dim
        self.device = device

        print(f"Encoder:\n \
                Encoder dimension: {self.image_emb_dim}")

        # Load pretrained ResNet-50 model and freeze its parameters
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Remove the last layer of ResNet-50
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Define a final classifier
        self.fc = nn.Linear(resnet.fc.in_features, self.image_emb_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            images (torch.Tensor): Input images of shape (BATCH, 3, 224, 224).

        Returns:
            torch.Tensor: Image features of shape (BATCH, IMAGE_EMB_DIM).
        """
        features = self.resnet(images)
        # Reshape features to (BATCH, 2048)
        features = features.reshape(features.size(0), -1).to(self.device)
        # Pass features through final linear layer
        features = self.fc(features).to(self.device)
        return features


class Decoder(nn.Module):
    """
    Decoder that uses an LSTM to generate captions from embedded words and encoded image features.
    The hidden and cell states of the LSTM are initialized using the encoded image features.

    Args:
        word_emb_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of the LSTM hidden state.
        num_layers (int): Number of LSTM layers.
        vocab_size (int): Size of the vocabulary (output dimension of the final linear layer).
        device (torch.device): Device to run the model on (CPU or GPU).
    """

    def __init__(self,
                 word_emb_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 vocab_size: int,
                 device: torch.device):
        super(Decoder, self).__init__()

        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = device

        # Initialize hidden and cell states
        self.hidden_state_0 = nn.Parameter(torch.zeros((self.num_layers, 1, self.hidden_dim)))
        self.cell_state_0 = nn.Parameter(torch.zeros((self.num_layers, 1, self.hidden_dim)))

        print(f"Decoder:\n \
                Embedding Size: {self.word_emb_dim},\n \
                LSTM Capacity: {self.hidden_dim},\n \
                Number of Layers: {self.num_layers},\n \
                Vocabulary Size: {self.vocab_size},\n \
                ")

        # Define LSTM layer
        self.lstm = nn.LSTM(self.word_emb_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=False)

        # Define final linear layer with LogSoftmax activation
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self,
                embedded_captions: torch.Tensor,
                hidden: torch.Tensor,
                cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the decoder.

        Args:
            embedded_captions (torch.Tensor): Embedded captions of shape (SEQ_LEN, BATCH, WORD_EMB_DIM).
            hidden (torch.Tensor): LSTM hidden state of shape (NUM_LAYER, BATCH, HIDDEN_DIM).
            cell (torch.Tensor): LSTM cell state of shape (NUM_LAYER, BATCH, HIDDEN_DIM).

        Returns:
            Tuple:
                - output (torch.Tensor): Output logits of shape (SEQ_LEN, BATCH, VOCAB_SIZE).
                - (hidden, cell) (Tuple[torch.Tensor, torch.Tensor]): Updated hidden and cell states.
        """
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded_captions, (hidden, cell))
        # Pass through final linear layer
        output = self.fc(output)
        return output, (hidden, cell)


def get_acc(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy of predictions compared to targets.

    Args:
        output (torch.Tensor): Model output of shape (BATCH, VOCAB_SIZE).
        target (torch.Tensor): Ground truth of shape (BATCH,).

    Returns:
        torch.Tensor: Accuracy as a scalar tensor.
    """
    probability = torch.exp(output)
    # Get the index of the maximum probability
    equality = (target == probability.max(dim=1)[1])
    return equality.float().mean()


if __name__ == '__main__':
    config = Config()

    encoder = Encoder(image_emb_dim=config.IMAGE_EMB_DIM, device=config.DEVICE)
    emb_layer = nn.Embedding(num_embeddings=config.VOCAB_SIZE,
                             embedding_dim=config.WORD_EMB_DIM,
                             padding_idx=0)
    decoder = Decoder(word_emb_dim=config.WORD_EMB_DIM,
                      hidden_dim=config.HIDDEN_DIM,
                      num_layers=config.NUM_LAYER,
                      vocab_size=config.VOCAB_SIZE,
                      device=config.DEVICE)

    encoder = encoder.to(config.DEVICE)
    emb_layer = emb_layer.to(config.DEVICE)
    decoder = decoder.to(config.DEVICE)

    # Create random tensor of images and captions
    images = torch.randn((32, 3, 256, 256)).to(config.DEVICE)
    captions = torch.randint(low=1, high=100, size=(32, 10), dtype=torch.int).to(config.DEVICE)

    # Pass images through encoder
    features = encoder(images=images)
    features = features.unsqueeze(0)
    print('Features size: ', features.size())  # (1, BATCH, IMAGE_EMB_DIM)

    # Initialize hidden and cell state
    hidden = features.repeat(config.NUM_LAYER, 1, 1)
    cell = features.repeat(config.NUM_LAYER, 1, 1)
    # hidden and cell: (NUM_LAYER, BATCH, HIDDEN_DIM)

    # Pass captions through embedding layer
    embedded_captions = emb_layer(captions)
    embedded_captions = embedded_captions.permute(1, 0, 2)
    print('Embedded captions size: ', embedded_captions.size())  # (SEQ_LEN, BATCH, WORD_EMB_DIM)

    # Pass embedded captions and features through decoder
    output, (hidden_state, cell_state) = decoder(embedded_captions=embedded_captions,
                                                 hidden=hidden,
                                                 cell=cell)

    print('Output size: ', output.size())        # (SEQ_LEN, BATCH, VOCAB_SIZE)
    print('Hidden size: ', hidden_state.size())  # (NUM_LAYER, BATCH, HIDDEN_DIM)
    print('Cell size: ', cell_state.size())      # (NUM_LAYER, BATCH, HIDDEN_DIM)
