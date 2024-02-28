import os
from collections import Counter

from nltk.tokenize import RegexpTokenizer
import argparse

from config import Config


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('text_file',
                        type=str,
                        help='Text file (in config.ROOT path) from which the vocabulary will be built')
    parser.add_argument('vocab_file',
                        type=str,
                        help='Text file (in config.ROOT path) in which the word2index from vocabulary will be saved')
    parser.add_argument('vocab_size',
                        type=int,
                        help='Size of vocabulary including the 4 predefined tokens')
    parsed_arguments = parser.parse_args()
    return parsed_arguments


class Vocab:
    """
    Offers word2index and index2word functionality after counting words in input sentences.
    Allows choosing the size of the vocabulary by taking the most common words. Explicitly reserves four indices:
    <pad>, <sos>, <eos> and <unk>.
    """
    def __init__(self, sentence_splitter=None):
        """
        Args:
        sentence_splitter: tokenizing function
        """
        self.config = Config()

        self.counter = Counter()
        self.word2index = dict()
        self.index2word = dict()

        self.size = 0

        # predefined tokens
        self.PADDING_INDEX = 0
        self.SOS = 1
        self.EOS = 2
        self.UNKNOWN_WORD_INDEX = 3

        if sentence_splitter is None:
            # matches sequences of characters including ones between < >
            word_regex = r'(?:\w+|<\w+>)'
            # tokenize the string into words
            sentence_splitter = RegexpTokenizer(word_regex).tokenize

        self.splitter = sentence_splitter

    def add_sentence(self, sentence: str):
        """
        Update word counts from sentence after tokenizing it into words
        """
        self.counter.update(self.splitter(sentence))

    def build_vocab(self, vocab_size: int, file_name: str):
        """
        Build vocabulary dictionaries word2index and index2word from a text file at config.ROOT path

        Args:
            - vocab_size (int): size of vocabulary (including 4 predefined tokens: <pad>, <sos>, <eos>, <unk>)
            - file_name (str): name of the text file from which the vocabulary will be built.
                Note: the lines in file are assumed to be in form: 'img_file_name COMMA caption'
                and it asssumes a header line (for example: 'captions.txt')
        """

        filepath = os.path.join(self.config.ROOT, file_name)

        with open(filepath) as file:
            for i, line in enumerate(file):

                # ignore header line
                if i == 0:
                    continue

                caption = line.strip().lower().split(",", 1)[1]  # id=0, caption=1
                vocab.add_sentence(caption)

        # adding predefined tokens in the vocabulary
        self.index2word[self.PADDING_INDEX] = '<pad>'
        self.word2index['<pad>'] = self.PADDING_INDEX
        self.index2word[self.SOS] = '<sos>'
        self.word2index['<sos>'] = self.SOS
        self.index2word[self.EOS] = '<eos>'
        self.word2index['<eos>'] = self.EOS
        self.index2word[self.UNKNOWN_WORD_INDEX] = '<unk>'
        self.word2index['<unk>'] = self.UNKNOWN_WORD_INDEX

        words = self.counter.most_common(vocab_size-4)

        # (index + 4) because first 4 tokens are the predefined ones
        for index, (word, _) in enumerate(words):
            self.word2index[word] = index+4
            self.index2word[index + 4] = word

        self.size = len(self.word2index)

    def word_to_index(self, word: str) -> int:
        """
        Map word to index from word2index dictionary in vocabulary

        Args:
            - word (str): word to be mapped

        Returns:
            - int: index matched to the word
        """
        try:
            return self.word2index[word]
        except KeyError:
            return self.UNKNOWN_WORD_INDEX

    def index_to_word(self, index: int) -> str:
        """
        Map word to index from index2word dictionary in vocabulary

        Args:
            - word (str): index to be mapped

        Returns:
            - str: word matched to the index
        """
        try:
            return self.index2word[index]
        except KeyError:
            return self.index2word[self.UNKNOWN_WORD_INDEX]

    def save_vocab(self, file_name: str):
        """
        Saves the word2index and index2word dictionaries in a text file at config.ROOT path.

        Args:
            file_name (str): name of the text file where the vocabulary will be saved (i.e 'word2index.txt')
                Note: the lines in file will be in form: 'word SPACE index'
        """

        filepath = os.path.join(self.config.ROOT, file_name)
        with open(filepath, 'a') as file:
            for word in self.word2index.keys():
                line = f"{word} {self.word2index[word]}\n"
                file.write(line)

    def load_vocab(self, file_name: str):
        """
        Load the word2index and index2word dictionaries from a text file at config.ROOT path

        Args:
            - file_name (str): name of the text file where the vocabulary is saved (i.e 'word2index.txt')
                Note: the lines in file are assumed to be in form: 'word SPACE index'
        """

        filepath = os.path.join(self.config.ROOT, file_name)

        self.word2index = dict()
        self.index2word = dict()

        with open(filepath) as file:
            for line in file:
                line = line.strip().split(' ')
                word, index = line[0], line[1]
                self.word2index[word] = int(index)
                self.index2word[int(index)] = word


# Build and save vocabulary
if __name__ == '__main__':

    args = parse_command_line_arguments()

    vocab = Vocab()
    vocab.build_vocab(args.vocab_size, str(args.text_file))
    vocab.save_vocab(str(args.vocab_file))
