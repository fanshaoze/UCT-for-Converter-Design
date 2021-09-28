import nltk
import json
import pickle
import argparse
from collections import Counter

# CAP_FILE = './data/captions/{}.tsv'
# DICT_OUTPUT_FILE = './data/captions/{}.json'

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def init_vocab(self):
        self.add_word('<pad>')
        # self.add_word('<start>')
        # self.add_word('<end>')
        # self.add_word('<and>')
        self.add_word('<unk>')


    def save(self, file_name):
        data = {}
        data['word2idx'] = self.word2idx
        data['idx2word'] = self.idx2word
        data['idx'] = self.idx
        import json
        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
        return

    def load(self, file_name):
        import json
        with open(file_name, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx = data['idx']
        return

def build_vocab(cap_file, threshold):
    """Build a simple vocabulary wrapper."""
    data = json.load(open(cap_file, 'r'))
    # with open(json, 'rb') as f:
    #     [data] = pickle.load(f)

    counter = Counter()
    print('total number of topologys',len(data))
    for instance in data:
        path_list = instance["paths"]
        counter.update(path_list)
    count = [len(x["paths"]) for x in data]
    print("mean length: ", sum(count)/len(count))
    print("max length: ", max(count))
    # for i in range(len(data)):
    #     captions = data[i]['captions']
    #     # for caption in captions:
    #     tokens = nltk.tokenize.word_tokenize(captions.lower())
    #     counter.update(tokens)

    #     if (i+1) % 1000 == 0:
    #         print("[{}/{}] Tokenized the captions.".format(i+1, len(data)))
            # break

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.init_vocab()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

def main(args):
    vocab = build_vocab(cap_file=args.data_set_path, threshold=args.threshold)
    vocab.save(args.save_output_path)
    print("Total vocabulary size: {}".format(len(vocab)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_path', type=str, default='dress')
    parser.add_argument('--save_output_path', type=str, default='dress')
    parser.add_argument('--threshold', type=int, default=1,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
