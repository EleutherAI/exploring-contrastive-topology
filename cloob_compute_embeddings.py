#!/usr/bin/env python3

"""Computes CLOOB embeddings (using the authors' CLOOB repo) of MS COCO 2017.

It requires the MS COCO 2017 training set in ./train2017 and the annotations in ./annotations.
The authors' CLOOB repo should be in ./cloob.
"""

import argparse
import json
import os
import sys

import numpy as np

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from webdataset import WebLoader, WebDataset

sys.path.append('./cloob-training')

from cloob_training import model_pt, pretrained

class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class TokenizerWrapper:
    def __init__(self, max_len=None):
        self.tokenizer = tokenizer.SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder['<start_of_text>']
        self.eot_token = self.tokenizer.encoder['<end_of_text>']
        self.context_length = 77
        self.max_len = self.context_length - 2 if max_len is None else max_len

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros([len(texts), self.context_length], dtype=torch.long)
        for i, text in enumerate(texts):
            tokens_trunc = self.tokenizer.encode(text)[:self.max_len]
            tokens = [self.sot_token, *tokens_trunc, self.eot_token]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, help="path to webdataset")
    p.add_argument('--batch-size', '-bs', type=int, default=100,
                   help='the batch size')
    p.add_argument('--config', type=str, default='cloob_laion_400m_vit_b_16_16_epochs',
                   help='the CLOOB model config')
    # p.add_argument('--val', action='store_true',
    #                help='use the validation set')
    p.add_argument('--output', type=str, required=True,
                   help='the output prefix')
    args = p.parse_args()

    device = torch.device('cuda:0')

    config = pretrained.get_config(args.config)
    clip_model = model_pt.get_pt_model(config)
    checkpoint = pretrained.download_checkpoint(config)
    clip_model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
    clip_model.eval().requires_grad_(False).to('cuda')

    clip_size = clip_model.config['image_encoder']['image_size']

    normalize = clip_model.normalize
    clip_tf = transforms.Compose([
        transforms.Resize(clip_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(clip_size),
        ToMode('RGB'),
        transforms.ToTensor(),
        normalize,
    ])


    def ttf(caption):
        if isinstance(caption, str):
            pass
        elif isinstance(caption, (list, tuple)):
            caption = caption[0]


        return clip_model.tokenize(caption, truncate=True).squeeze(0)


    # prefix = 'val' if args.val else 'train'
    prefix = 'train'

    dataset = WebDataset(args.dataset).decode("pil").map_dict(jpg=clip_tf, txt=ttf).to_tuple("jpg", "txt")

    loader = WebLoader(dataset, args.batch_size, num_workers=20)

    image_embeds, text_embeds = [], []

    for images, texts in tqdm(loader):
        image_embeds_batch = F.normalize(clip_model.image_encoder(images.to(device)).float(), dim=1)
        text_embeds_batch = F.normalize(clip_model.text_encoder(texts.to(device)).float(), dim=-1)
        image_embeds.append(image_embeds_batch.cpu())
        text_embeds.append(text_embeds_batch.cpu())

    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    # obj = {'image_embeds': image_embeds,
    #        'text_embeds': text_embeds}

    # torch.save(obj, args.output)

    np.save(f'{args.output}_image_embeds.npy', image_embeds.numpy())
    np.save(f'{args.output}_text_embeds.npy', text_embeds.numpy())


if __name__ == '__main__':
    main()
