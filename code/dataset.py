import os

import torch
import torch.utils.data
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from vocab import Vocab


def preprocessing_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def denormalize(image: torch.Tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()


class ImageCaptionDataset():

    def __init__(self,
                 captions_text_file: str,
                 vocab: Vocab, image_folder: str,
                 transform=preprocessing_transforms()):
        """
        Holds a list of samples where each sample is a dictionary containing the image file ID and
        the caption of that image as a list of word indices.

        Args:
            - captions_text_file (str): text file of captions in config.ROOT path.
                Note: the lines in file are assumed to be in form:
                'img_file COMMA caption' and it asssumes a header line
            - vocab (Vocab): vocabulary object with built vocabulary for splitting the captions
            and performing word2index()
            - image_folder (str): image folder in config.ROOT containing the images corresponding
            to captions_text_file
            - transform (Compose, optional): transform operations to be applied to images.
            Defaults to preprocessing_transforms()
        """

        self.config = Config()

        self.samples = []

        self.images_root = os.path.join(self.config.ROOT, image_folder)
        self.transform = transform

        sample_list_path = os.path.join(self.config.ROOT, captions_text_file)
        with open(file=sample_list_path, mode='r', encoding='utf-8') as file:
            for i, line in enumerate(file):

                # skip header line
                if i == 0:
                    continue

                # sample: img_file COMMA caption
                sample = line.strip().lower().split(",", 1)

                image_id = sample[0]
                caption = sample[1]

                caption = '<sos> ' + caption + ' <eos>'

                # tokenize the caption into words
                words = vocab.splitter(caption)

                # map the words into indices and save in a list
                word_ids = [vocab.word_to_index(word) for word in words]

                sample = {
                    "image_id": image_id,
                    "caption": torch.LongTensor(word_ids)
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = os.path.join(self.images_root, sample['image_id'])

        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"WARNING: Could not find image '{img_name}'. ")
            image = Image.new('RGB', (256, 256))

        if self.transform:
            image = self.transform(image)

        return image, sample['caption']


class Padding:

    def __init__(self, pad_idx: int, batch_first=True):
        """ When called in DataLoader it returns batched images with batched padded captions

        Args:
            - pad_idx (int): value of padding index to be used for padding the captions
            - batch_first (bool, optional): Defaults to True.
        """
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        # imgs: (3, 224, 224) -> (1, 3, 224, 224)

        imgs = torch.cat(imgs, dim=0)
        # imgs: (BATCH, 3, 224, 224)

        captions = [item[1] for item in batch]
        # captions: [BATCH, SEQ_LENGTH]

        captions = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, captions


def get_data_loader(dataset, batch_size=32, pad_index=0):
    """
    Args:
        - dataset (ImageCaptionDataset)
        - batch_size (int, optional):  Defaults to 32.
        - pad_index (int, optional):  Defaults to 0.

    Returns:
        Batched loader of images and captions
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=3,
        pin_memory=True,
        shuffle=True,
        collate_fn=Padding(pad_idx=pad_index, batch_first=True)
    )


def sort_captions(input_file, output_file):
    input_file = open(input_file, 'r')
    lines = input_file.readlines()

    header = lines[0]
    lines = lines[1:]

    sorted_lines = sorted(lines, key=lambda x: len(x.split(',', 1)[1]))
    output_file = open(output_file, 'w')
    output_file.write(header)
    for line in sorted_lines:
        output_file.write(line)

    input_file.close()
    output_file.close()


if __name__ == '__main__':

    config = Config()

    print('Loading vocabulary...')
    vocab = Vocab()
    vocab.load_vocab(config.VOCAB_FILE)

    # sort_captions('train_list.txt', 'train_list_sorted.txt')
    # sort_captions('val_list.txt', 'val_list_sorted.txt')

    print('Initializing dataset objects...')
    train_data = ImageCaptionDataset('train_list.txt', vocab, 'images', transform=preprocessing_transforms())

    print('Getting data loader...')
    train_loader = get_data_loader(train_data, batch_size=config.BATCH, pad_index=0)

    x, y = next(iter(train_loader))
    # x (img) : [batch, 3, 224, 224]
    # y (caption): [batch, seq_length]

    import matplotlib.pyplot as plt

    for image, caption in zip(x, y):
        image = denormalize(image)
        caption = [vocab.index_to_word(int(word_id)) for word_id in caption]
        caption = ' '.join(word for word in caption)
        plt.imshow(image)
        plt.title(caption)
        plt.show()
        plt.pause(1)
