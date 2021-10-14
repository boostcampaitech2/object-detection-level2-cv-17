
# MMDtection

## Requirements

```
pip install -r requirements
```

## Usage

### Train
```
$ python train.py -h
usage: train.py [-h] [--seed [SEED]] [--batch_size [BATCH_SIZE]]
                [--model_build_type [MODEL_BUILD_TYPE]] [--validate]
                [--no-validate] [--wandb] [--no-wandb]
                [--max_keep_ckpts [MAX_KEEP_CKPTS]]
                [--ckpts_interval [CKPTS_INTERVAL]] [--ckpt_name [CKPT_NAME]]
                [--data_dir [DATA_DIR]] [--config_dir [CONFIG_DIR]]
                [--config_file [CONFIG_FILE]]

optional arguments:
  -h, --help            show this help message and exit
  --seed [SEED]         random seed (default: 1995)
  --batch_size [BATCH_SIZE]
                        batch size (default: 2)
  --model_build_type [MODEL_BUILD_TYPE]
                        model_build_type (default: 0) 0(init_weights),
                        1(load_checkpoint)
  --validate            validation
  --no-validate         no validation
  --wandb               wandb
  --no-wandb            not use wandb
  --max_keep_ckpts [MAX_KEEP_CKPTS]
                        max_keep_ckpts (default: 5)
  --ckpts_interval [CKPTS_INTERVAL]
                        ckpts_interval (default: 1)
  --ckpt_name [CKPT_NAME]
  --data_dir [DATA_DIR]
  --config_dir [CONFIG_DIR]
  --config_file [CONFIG_FILE]
```
### Inference
```
$ python inference.py -h
usage: inference.py [-h] [--seed [SEED]] [--batch_size [BATCH_SIZE]]
                    [--ckpt_name [CKPT_NAME]] [--data_dir [DATA_DIR]]
                    [--config_dir [CONFIG_DIR]] [--config_file [CONFIG_FILE]]

optional arguments:
  -h, --help            show this help message and exit
  --seed [SEED]         random seed (default: 1995)
  --batch_size [BATCH_SIZE]
                        batch size (default: 2)
  --ckpt_name [CKPT_NAME]
  --data_dir [DATA_DIR]
  --config_dir [CONFIG_DIR]
  --config_file [CONFIG_FILE]
```

## Running

### Train
```
$ python train.py [--seed [SEED]] [--batch_size [BATCH_SIZE]]
                  [--model_build_type [MODEL_BUILD_TYPE]] 
                  [--validate or --no-validate] [--wandb or --no-wandb]
                  [--max_keep_ckpts [MAX_KEEP_CKPTS]]
                  [--ckpts_interval [CKPTS_INTERVAL]] [--ckpt_name [CKPT_NAME]]
                  [--data_dir [DATA_DIR]] [--config_dir [CONFIG_DIR]]
                  [--config_file [CONFIG_FILE]]
```
### Inference
```
python inference.py [--seed [SEED]] [--batch_size [BATCH_SIZE]]
                    [--ckpt_name [CKPT_NAME]] [--data_dir [DATA_DIR]]
                    [--config_dir [CONFIG_DIR]] [--config_file [CONFIG_FILE]]
```