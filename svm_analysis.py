#!/usr/bin/env python3

"""Analyzes paired contrastive embeddings with an SVM."""

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm


class SVMAnalysis:
    def __init__(self, image_embeds_path, text_embeds_path):
        self.image_embeds = np.load(image_embeds_path)
        self.image_embeds /= np.sum(self.image_embeds ** 2, axis=1, keepdims=True)
        self.text_embeds = np.load(text_embeds_path)
        self.text_embeds /= np.sum(self.text_embeds ** 2, axis=1, keepdims=True)
        self.dataset_x = np.concatenate([self.image_embeds, self.text_embeds])
        self.dataset_y = np.concatenate([np.zeros([len(self.image_embeds)]), np.ones([len(self.text_embeds)])])
        self.svm = sklearn.svm.LinearSVC()
        self.svm.fit(self.dataset_x, self.dataset_y)

    def get_accuracy(self):
        return np.sum(self.svm.predict(self.dataset_x) == self.dataset_y) / len(self.dataset_x)

    def save_histogram(self, path):
        coef_norm = np.linalg.norm(self.svm.coef_)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(self.svm.decision_function(self.dataset_x[:len(self.image_embeds)]) / coef_norm, bins=100, density=True)
        ax.hist(self.svm.decision_function(self.dataset_x[len(self.image_embeds):]) / coef_norm, bins=100, density=True)
        fig.savefig(path)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image_embeds', type=str,
                   help='the image embeddings path')
    p.add_argument('text_embeds', type=str,
                   help='the text embeddings path')
    p.add_argument('--output-prefix', type=str, default='out',
                   help='the output prefix')
    args = p.parse_args()

    matplotlib.use('Agg')

    analysis = SVMAnalysis(args.image_embeds, args.text_embeds)
    print('Accuracy:', analysis.get_accuracy())
    analysis.save_histogram(f'{args.output_prefix}_hist.png')


if __name__ == '__main__':
    main()
