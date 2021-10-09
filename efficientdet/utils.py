# 라이브러리 및 모듈 import
import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from collections import defaultdict

from pathlib import Path
import glob
import re

# stratified Group K-fold cross validation
def stratified_split(annotation, index, IsValid):
    coco = COCO(annotation)

    df = pd.DataFrame(coco.dataset['annotations'])
    X = df['id']              # 객체 번호 [0~23143]
    y = df['category_id']     # 객체 당 카테고리 번호 [0~9]
    groups = df['image_id']   # 이미지 번호 [0~4882]
    seed = 777
    k = 5
    
    labels_num = y.max() + 1
    # https://stackoverflow.com/a/39132900/14019325
    # 기존 코드의 첫번째 loop와 동일합니다. 각 image 별 label 개수를 확인합니다.
    y_counts_per_group = df.groupby(['image_id', 'category_id']).size().unstack(fill_value=0)
    y_counts_per_fold = np.zeros((k, labels_num))

    # scale을 미리 계산하여 연산을 줄입니다.
    y_norm_counts_per_group = y_counts_per_group / y_counts_per_group.sum()
    # suffle & sort
    shuffled_and_sorted_index = y_norm_counts_per_group.sample(frac=1, random_state=seed).std(axis=1).sort_values(ascending=False).index
    y_norm_counts_per_group = y_norm_counts_per_group.loc[shuffled_and_sorted_index]

    groups_per_fold = defaultdict(set)

    for g, y_counts in zip(y_norm_counts_per_group.index, y_norm_counts_per_group.values):
        best_fold = None
        min_eval = None
        for fold_i in range(k):
            # 기존 코드 eval_y_counts_per_fold 와 동일합니다.
            y_counts_per_fold[fold_i] += y_counts
            fold_eval = y_counts_per_fold.std(axis=0).mean()  # numpy를 활용하여 연산을 단순화 합니다.
            y_counts_per_fold[fold_i] -= y_counts
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fold_i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    train_groups = all_groups - groups_per_fold[index]
    valid_groups = groups_per_fold[index] 

    if IsValid:
        return list(valid_groups), 'valid'
    else:
        return list(train_groups), 'train'


# loss 추적
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"