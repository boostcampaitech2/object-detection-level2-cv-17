
Splits COCO annotations file into train sets

## Requirements

```
pip install -r requirements
```

## Usage
```
$ python cocosplit.py -h
usage: StratifiedGroupKFold.py [-h] [--anns ANNS] [--train TRAIN] [--valid VALID] [--seed SEED]
                               [--k K]

Splits COCO annotations file into train sets.

optional arguments:
  -h, --help     show this help message and exit
  --anns ANNS    Path to COCO annotations file.(*.json)
  --train TRAIN  Where to store COCO training annotations
  --valid VALID  Where to store COCO test annotations
  --seed SEED    Random seed
  --k K          K times split
```

## Running
```
$ python StratifiedGroupKFold.py [--anns ANNS] [--train TRAIN] [--valid VALID] [--seed SEED]
```

### default  
--anns  = './train.json'  
--train = 'train_split'  
--valid = 'valid_split'  
--seed  = 1995  
--k     = 5  