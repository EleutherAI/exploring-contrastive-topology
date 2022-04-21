#!/usr/bin/env python3

"""Computes CLOOB embeddings (using the authors' CLOOB repo) of MS COCO 2017.

It requires the MS COCO 2017 training set in ./train2017 and the annotations in ./annotations.
The authors' CLOOB repo should be in ./cloob and the 'pycocotools' pip package must also be installed.
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

sys.path.append('./cloob/src/clip')

import tokenizer
from model import CLIPGeneral


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
    p.add_argument('--batch-size', '-bs', type=int, default=100, 
                   help='the batch size')
    p.add_argument('--config', type=str, default='cloob/src/training/model_configs/RN50.json',
                   help='the CLOOB model config')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='the CLOOB model checkpoint')
    # p.add_argument('--val', action='store_true',
    #                help='use the validation set')
    p.add_argument('--output', type=str, required=True,
                   help='the output prefix')
    args = p.parse_args()

    device = torch.device('cuda:0')

    model_info = json.load(open(args.config))
    model_info['method'] = 'cloob'
    clip_model = CLIPGeneral(**model_info).to(device).eval().requires_grad_(False)
    ckpt = torch.load(args.checkpoint)
    sd = {k[len('module.'):]: v for k, v in ckpt['state_dict'].items()}
    res = clip_model.load_state_dict(sd, strict=False)
    print(res)
    del ckpt, sd
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    clip_tf = transforms.Compose([
        transforms.Resize(clip_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(clip_size),
        ToMode('RGB'),
        transforms.ToTensor(),
        normalize,
    ])

    tok_wrap = TokenizerWrapper()
    def ttf(caption):
        return tok_wrap(caption[0]).squeeze(0)

    # prefix = 'val' if args.val else 'train'
    prefix = 'train'

    dataset = datasets.CocoCaptions(f'{prefix}2017',
                                    f'annotations/captions_{prefix}2017.json',
                                    transform=clip_tf,
                                    target_transform=ttf)
    loader = data.DataLoader(dataset, args.batch_size, num_workers=22)

    image_embeds, text_embeds = [], []

    for images, texts in tqdm(loader):
        image_embeds_batch = F.normalize(clip_model.encode_image(images.to(device)).float(), dim=1)
        text_embeds_batch = F.normalize(clip_model.encode_text(texts.to(device)).float(), dim=-1)
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
