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
    """
    Define preprocessing transformations for images.

    Returns:
        torchvision.transforms.Compose: Composed transformations for resizing, cropping, 
        converting to tensor, and normalizing images.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def denormalize(image: torch.Tensor):
    """
    Denormalize a tensor image back to its original pixel values.

    Args:
        image (torch.Tensor): Tensor image to be denormalized.

    Returns:
        numpy.ndarray: Denormalized image as a numpy array with pixel values in range [0, 255].
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()


class ImageCaptionDataset(torch.utils.data.Dataset):
    """
    Dataset class that provides image-caption pairs.

    Args:
        captions_text_file (str): Path to the text file containing captions. 
                                  Each line should be in the format 'img_file,caption'.
        vocab (Vocab): Vocabulary object used for tokenizing captions and converting words to indices.
        image_folder (str): Path to the folder containing images corresponding to the captions.
        transform (callable, optional): Transformations to be applied to images. Defaults to preprocessing_transforms().
    """

    def __init__(self,
                 captions_text_file: str,
                 vocab: Vocab,
                 image_folder: str,
                 transform=preprocessing_transforms()):
        self.config = Config()
        self.samples = []
        self.images_root = os.path.join(self.config.ROOT, image_folder)
        self.transform = transform

        sample_list_path = os.path.join(self.config.ROOT, captions_text_file)
        with open(file=sample_list_path, mode='r', encoding='utf-8') as file:
            for line in file:
                # Parse each line to extract image_id and caption
                sample = line.strip().lower().split(",", 1)
                image_id = sample[0]
                caption = sample[1]

                caption = '<sos> ' + caption + ' <eos>'

                # Tokenize the caption into words and convert to indices
                words = vocab.splitter(caption)
                word_ids = [vocab.word_to_index(word) for word in words]

                self.samples.append({
                    "image_id": image_id,
                    "caption": torch.LongTensor(word_ids)
                })

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, caption) where `image` is a tensor and `caption` is a tensor of word indices.
        """
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
    """
    Custom collate function to pad captions to the same length in a batch.

    Args:
        pad_idx (int): Padding index to be used for padding the captions.
        batch_first (bool, optional): If True, the output tensors will have shape (batch, seq_length).
                                       Defaults to True.
    """

    def __init__(self, pad_idx: int, batch_first=True):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)  # Shape: (BATCH, 3, 224, 224)

        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs, captions


def get_data_loader(dataset, batch_size=32, pad_index=0, shuffle=False):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (ImageCaptionDataset): Dataset to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        pad_index (int, optional): Index used for padding captions. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

    Returns:
        DataLoader: DataLoader instance for batching the dataset.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=3,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=Padding(pad_idx=pad_index, batch_first=True)
    )


def sort_captions(input_file, output_file):
    """
    Sort captions by length and save to a new file.

    Args:
        input_file (str): Path to the input file with unsorted captions.
        output_file (str): Path to the output file where sorted captions will be saved.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    sorted_lines = sorted(lines, key=lambda x: len(x.split(',', 1)[1]))
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in sorted_lines:
            outfile.write(line)


if __name__ == '__main__':
    config = Config()

    print('Loading vocabulary...')
    vocab = Vocab()
    vocab.load_vocab(config.VOCAB_FILE)

    sort_captions('train_list.txt', 'train_list_sorted.txt')
    sort_captions('val_list.txt', 'val_list_sorted.txt')

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
