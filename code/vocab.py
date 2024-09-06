import os
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import argparse
from config import Config


def parse_command_line_arguments():
    """
    Parse command line arguments for building and saving vocabulary.

    Returns:
        argparse.Namespace: Parsed arguments containing 'text_file', 'vocab_file', and 'vocab_size'.
    """
    parser = argparse.ArgumentParser(description='Build vocabulary from a text file and save it.')
    parser.add_argument(
        'text_file',
        type=str,
        help='Text file (in config.ROOT path) from which the vocabulary will be built'
    )
    parser.add_argument(
        'vocab_file',
        type=str,
        help='Text file (in config.ROOT path) in which the word2index from vocabulary will be saved'
    )
    parser.add_argument(
        'vocab_size',
        type=int,
        help='Size of vocabulary including the 4 predefined tokens'
    )
    parsed_arguments = parser.parse_args()
    return parsed_arguments


class Vocab:
    """
    Provides word2index and index2word functionality after counting words in input sentences.
    Allows choosing the size of the vocabulary by taking the most common words. 
    Explicitly reserves four indices: <pad>, <sos>, <eos>, and <unk>.
    """

    def __init__(self, sentence_splitter=None):
        """
        Initialize the Vocab class.

        Args:
            sentence_splitter (callable, optional): Function to tokenize sentences. If None, a default tokenizer is used.
        """
        self.config = Config()

        self.counter = Counter()
        self.word2index = dict()
        self.index2word = dict()
        self.size = 0

        # Predefined tokens
        self.PADDING_INDEX = 0
        self.SOS = 1
        self.EOS = 2
        self.UNKNOWN_WORD_INDEX = 3

        if sentence_splitter is None:
            # Tokenize words and special tokens like <sos> using a regular expression
            word_regex = r'(?:\w+|<\w+>)'
            sentence_splitter = RegexpTokenizer(word_regex).tokenize

        self.splitter = sentence_splitter

    def add_sentence(self, sentence: str):
        """
        Tokenize a sentence and update word counts.

        Args:
            sentence (str): Input sentence to be tokenized and counted.
        """
        self.counter.update(self.splitter(sentence))

    def build_vocab(self, vocab_size: int, file_name: str):
        """
        Build vocabulary from a text file and store the word2index and index2word dictionaries.

        Args:
            vocab_size (int): Size of the vocabulary, including the 4 predefined tokens.
            file_name (str): Name of the text file to build the vocabulary from.
                             The file should contain lines in the format 'img_file_name,caption'
                             with the first line as a header.
        """
        filepath = os.path.join(self.config.ROOT, file_name)

        with open(filepath) as file:
            for i, line in enumerate(file):
                # Skip the header line
                if i == 0:
                    continue
                # Extract caption and add it to the vocabulary
                caption = line.strip().lower().split(",", 1)[1]
                self.add_sentence(caption)

        # Add predefined tokens to the vocabulary
        self.index2word[self.PADDING_INDEX] = '<pad>'
        self.word2index['<pad>'] = self.PADDING_INDEX
        self.index2word[self.SOS] = '<sos>'
        self.word2index['<sos>'] = self.SOS
        self.index2word[self.EOS] = '<eos>'
        self.word2index['<eos>'] = self.EOS
        self.index2word[self.UNKNOWN_WORD_INDEX] = '<unk>'
        self.word2index['<unk>'] = self.UNKNOWN_WORD_INDEX

        # Add the most common words to the vocabulary, excluding the predefined tokens
        words = self.counter.most_common(vocab_size - 4)

        for index, (word, _) in enumerate(words):
            self.word2index[word] = index + 4
            self.index2word[index + 4] = word

        self.size = len(self.word2index)

    def word_to_index(self, word: str) -> int:
        """
        Retrieve the index of a word from the vocabulary.

        Args:
            word (str): Word to retrieve the index for.

        Returns:
            int: The index of the word. Returns the index for <unk> if the word is not found.
        """
        return self.word2index.get(word, self.UNKNOWN_WORD_INDEX)

    def index_to_word(self, index: int) -> str:
        """
        Retrieve the word corresponding to an index from the vocabulary.

        Args:
            index (int): Index to retrieve the word for.

        Returns:
            str: The word corresponding to the index. Returns '<unk>' if the index is not found.
        """
        return self.index2word.get(index, '<unk>')

    def save_vocab(self, file_name: str):
        """
        Save the word2index and index2word dictionaries to a text file.

        Args:
            file_name (str): Name of the text file to save the vocabulary to.
                             The file will contain lines in the format 'word index'.
        """
        filepath = os.path.join(self.config.ROOT, file_name)
        with open(filepath, 'w') as file:
            for word, index in self.word2index.items():
                file.write(f"{word} {index}\n")

    def load_vocab(self, file_name: str):
        """
        Load the word2index and index2word dictionaries from a text file.

        Args:
            file_name (str): Name of the text file containing the vocabulary.
                             The file should contain lines in the format 'word index'.
        """
        filepath = os.path.join(self.config.ROOT, file_name)

        self.word2index.clear()
        self.index2word.clear()

        with open(filepath) as file:
            for line in file:
                word, index = line.strip().split()
                index = int(index)
                self.word2index[word] = index
                self.index2word[index] = word


# Build and save vocabulary
if __name__ == '__main__':
    args = parse_command_line_arguments()

    vocab = Vocab()
    vocab.build_vocab(args.vocab_size, args.text_file)
    vocab.save_vocab(args.vocab_file)
