import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import nltk
from PIL import Image
import json
import numpy as np

# SEQ_LEN = None
class Dataset(data.Dataset):

    def __init__(self, data_file_name, vocab, transform=None, max_seq_len=64, label_len=5):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            transform: image transformer.
            vocab: pre-processed vocabulary.
        """
        # self.root = root
        with open(data_file_name, 'r') as f:
            self.data = json.load(f)
        self.ids = range(len(self.data))
        self.vocab = vocab
        # self.transform = transform
        # self.return_target = return_target
        self.seq_len = max_seq_len
        # SEQ_LEN = max_seq_len
        # self.label_len = label_len

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        
        path_list = []
        paths = data[id]['paths']
        # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption_texts).lower()) 

        if len(paths) >= self.seq_len:
            paths = paths[:self.seq_len]
                
        # caption.append(vocab('<start>'))
        path_list.extend([vocab(token) for token in paths])
        # caption.append(vocab('<end>'))
        path_list = torch.Tensor(path_list)

        eff = torch.Tensor([data[id]['eff']]).float()
        vout = torch.Tensor([data[id]['vout']]).float()

        return path_list, eff, vout

    def __len__(self):
        return len(self.ids)

    def append_data(self, path_set, effs, vouts):
        for paths, eff, vout in zip(path_set, effs, vouts):
            self.data.append({'paths': paths, 'eff': eff, 'vout': vout})

        self.ids = range(len(self.data))


def getRawData(data_path,vocab,max_seq_len):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data_x = []

    for item in data:
        paths = item['paths']
        if len(paths) >= max_seq_len:
            paths = paths[:max_seq_len]
        data_x.append([vocab(x) for x in paths])

    data_x = torch.Tensor(data_x)

    return data_x[:500]

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    path_list, effs, vouts = zip(*data)


    # Merge images (from tuple of 3D tensor to 4D tensor).
    #path_list = torch.stack(path_list, 0)
    # image1 = torch.stack(image1, 0)
    
    effs = torch.stack(effs, 0)
    vouts = torch.stack(vouts, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in path_list]

    path_tensor = torch.zeros(len(path_list), max(lengths)).long()
    # captions_tgt = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(path_list):
        end = lengths[i]
        path_tensor[i, :end] = cap[:end]
        # captions_tgt[i, :end-1] = cap[1:end]
    padding_mask = (path_tensor != 0)

    #assert padding_mask.all()
    #assert (path_tensor == path_list[0].long()).all()
    return path_tensor, effs, vouts, padding_mask

# def collate_fn_test(data):
#     """Creates mini-batch tensors from the list of tuples (image, caption).
#     Args:
#         data: list of tuple (image, caption). 
#             - image: torch tensor of shape
#             - caption: torch tensor of shape (?); variable length.
#     Returns:
#         images: torch tensor of images.
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     image0, image1, _, image0_label, image1_label = zip(*data)
#     # Merge images (from tuple of 3D tensor to 4D tensor).
#     image0 = torch.stack(image0, 0)
#     image1 = torch.stack(image1, 0)

#     image0_label = torch.stack(image0_label, 0)
#     image1_label = torch.stack(image1_label, 0)
#     # # Merge captions (from tuple of 1D tensor to 2D tensor).
#     # lengths = [len(cap) for cap in captions]
#     # captions_src = torch.zeros(len(captions), max(lengths)).long()
#     # captions_tgt = torch.zeros(len(captions), max(lengths)).long()
#     # for i, cap in enumerate(captions):
#     #     end = lengths[i]
#     #     captions_src[i, :end-1] = cap[:end-1]
#     #     captions_tgt[i, :end-1] = cap[1:end]
#     # # caption_padding_mask = (captions_src != 0)
#     # return target_images, candidate_images, captions_src, captions_tgt
#     return image0, image1, image0_label, image1_label


def get_loader(data, vocab, transform, batch_size, shuffle, num_workers=1, max_seq_len=64, attribute_len=5):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    # relative caption dataset
    if type(data) == str:
        print('Reading data from', data)
        dataset = Dataset(
                          data_file_name=data,
                          vocab=vocab,
                          transform=transform,
                          max_seq_len=max_seq_len,
                          label_len=attribute_len
                          )
    else:
        # use preloaded data
        dataset = data

    print('data size',len(dataset))
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              timeout=60)

    return data_loader

# def get_loader_test(data_file_path, vocab, transform, batch_size, shuffle, num_workers=1,max_seq_len=64,attribute_len=5):
#     """Returns torch.utils.data.DataLoader for custom dataset."""
#     # relative caption dataset
#     print('Reading data from',data_file_path)
#     dataset = Dataset(
#                       data_file_name=data_file_path,
#                       vocab=vocab,
#                       transform=transform,
#                       max_seq_len=max_seq_len,
#                       label_len=attribute_len
#                       )
#     print('data size',len(dataset))
#     # Data loader for the dataset
#     # This will return (images, captions, lengths) for each iteration.
#     # images: a tensor of shape (batch_size, 3, 224, 224).
#     # captions: a tensor of shape (batch_size, padded_length).
#     # lengths: a list indicating valid length for each caption. length is (batch_size)
#     data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=1,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               collate_fn=collate_fn_test,
#                                               timeout=60)
#     return data_loader

def load_ori_token_data(data_file_name):
    test_data_captions = []
    with open(data_file_name, 'r') as f:
        data = json.load(f)

        for line in data:
            caption_texts = line['captions']
            temp = []
            for c in caption_texts:
                # tokens = nltk.tokenize.word_tokenize(str(c).lower())
                temp.append(c)
            test_data_captions.append(temp)

    
    return test_data_captions


def load_ori_token_data_new(data_file_name):
    test_data_captions = {}
    with open(data_file_name, 'r') as f:
        data = json.load(f)
        count = 0
        for line in data:
            caption_texts = line['captions']
            caption_texts = ["it " + x for x in caption_texts]
            # temp = []
            # for c in caption_texts:
            #     # tokens = nltk.tokenize.word_tokenize(str(c).lower())
            #     temp.append(c)
            test_data_captions[count] = caption_texts
            count += 1
    
    return test_data_captions
