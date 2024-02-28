import os

import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.utils.data
from PIL import Image
import argparse

from dataset import denormalize
from vocab import Vocab
from model import Decoder, Encoder
from config import Config
from test_show import generate_caption


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image_file', type=str, help='An image to caption from your config.ROOT path')
    parsed_arguments = parser.parse_args()
    return parsed_arguments


if __name__ == '__main__':

    args = parse_command_line_arguments()

    config = Config()

    print('Loading vocabulary...')
    vocab = Vocab()
    vocab.load_vocab(config.VOCAB_FILE)

    print('Transforming image...')
    img_name = os.path.join(config.ROOT, str(args.image_file))
    image = Image.open(img_name).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    image = transform(image)

    print('Creating model...')
    image_encoder = Encoder(image_emb_dim=config.IMAGE_EMB_DIM,
                            device=config.DEVICE)
    emb_layer = torch.nn.Embedding(num_embeddings=config.VOCAB_SIZE,
                                   embedding_dim=config.WORD_EMB_DIM,
                                   padding_idx=vocab.PADDING_INDEX)
    image_decoder = Decoder(image_emb_dim=config.IMAGE_EMB_DIM,
                            word_emb_dim=config.WORD_EMB_DIM,
                            hidden_dim=config.HIDDEN_DIM,
                            num_layers=config.NUM_LAYER,
                            vocab_size=config.VOCAB_SIZE,
                            device=config.DEVICE)

    emb_layer.eval()
    image_encoder.eval()
    image_decoder.eval()

    LOAD_WEIGHTS = True
    if LOAD_WEIGHTS:
        print("Loading pretrained weights...")
        emb_layer.load_state_dict(torch.load(config.EMBEDDING_WEIGHT_FILE))
        image_encoder.load_state_dict(torch.load(config.ENCODER_WEIGHT_FILE))
        image_decoder.load_state_dict(torch.load(config.DECODER_WEIGHT_FILE))

    emb_layer = emb_layer.to(config.DEVICE)
    image_encoder = image_encoder.to(config.DEVICE)
    image_decoder = image_decoder.to(config.DEVICE)
    image = image.to(config.DEVICE)

    print('Generating caption...')
    sentence = generate_caption(image, image_encoder, emb_layer, image_decoder, vocab, device=config.DEVICE)
    sentence = [vocab.index2word[vocab.SOS]] + sentence
    sentence = ' '.join(word for word in sentence)

    image = denormalize(image.cpu())
    plt.imshow(image)
    plt.title(sentence)
    plt.show()
    plt.pause(1)
