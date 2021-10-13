import random
import argparse
import numpy as np
import pandas as pd
import json
from pandas import json_normalize
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description='Splits COCO annotations file into train sets.')
parser.add_argument('--anns', type=str, default='./train.json', help='Path to COCO annotations file.(*.json)')
parser.add_argument('--train', type=str, default="train_split", help='Where to store COCO training annotations')
parser.add_argument('--valid', type=str, default="valid_split", help='Where to store COCO test annotations')
parser.add_argument('--seed', type=int, default=1995, help='Random seed')
parser.add_argument('--k', type=int, default=5, help='K times split')

args = parser.parse_args()

def get_json_data(args):
    with open(args.anns, 'r') as f:
        json_data = json.load(f)
    df = json_normalize(json_data['annotations'])
    categories = json_data['categories']
    info = json_data['info']
    licenses = json_data['licenses']
    images = json_normalize(json_data['images'])

    return df, categories, info, licenses, images

def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def main(args):
    seed = args.seed
    k = args.k

    df, categories, info, licenses, images = get_json_data(args)

    x = df['id']
    y = df['category_id']
    groups = df['image_id']

    labels_num = y.max() + 1

    y_counts_per_group = df.groupby(['image_id', 'category_id']).size().unstack(fill_value=0)
    y_counts_per_fold = np.zeros((k, labels_num))

    y_norm_counts_per_group = y_counts_per_group / y_counts_per_group.sum()

    shuffled_and_sorted_index = y_norm_counts_per_group.sample(frac=1, random_state=seed).std(axis=1).sort_values(ascending=False).index
    y_norm_counts_per_group = y_norm_counts_per_group.loc[shuffled_and_sorted_index]

    groups_per_fold = defaultdict(set)

    for g, y_counts in zip(y_norm_counts_per_group.index, y_norm_counts_per_group.values):
        best_fold = None
        min_eval = None
        for fold_i in range(k):
            y_counts_per_fold[fold_i] += y_counts
            fold_eval = y_counts_per_fold.std(axis=0).mean()
            y_counts_per_fold[fold_i] -= y_counts
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fold_i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)

    distrs = [get_distribution(y)]
    index = ['training set']

    # Save K times
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_df = df.loc[df['image_id'].isin(train_groups)]
        valid_df = df.loc[df['image_id'].isin(test_groups)]

        train_images = images[images['id'].isin(train_df['id'].unique())]
        valid_images = images[images['id'].isin(valid_df['id'].unique())]

        save_coco(f'{args.train}_{i}.json', info, licenses, train_images.to_dict('records'), train_df.to_dict('records'), categories)
        save_coco(f'{args.valid}_{i}.json', info, licenses, valid_images.to_dict('records'), valid_df.to_dict('records'), categories)

if __name__ == "__main__":
    main(args)