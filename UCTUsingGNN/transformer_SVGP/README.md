(1) Transformer only no gp													RSE:	0.031050

```
python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/run2.log -save_model=save_model/model2.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=100 -n_layers=3 -d_model=128
```

(2) End2End training fixed pretrained Transformer + variational GP			RSE:	0.029211

```
python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/batch_gp.log -save_model=save_model/batch_gp.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=500 -n_layers=1 -d_model=128 -pretrained_model=save_model/model.pt
在train_epoch()里加上with torch.no_grad():
```

(3) End2End training random initialed transformer + variational GP			RSE:	0.020594

```
python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/batch_gp.log -save_model=save_model/batch_gp.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=700 -n_layers=1 -d_model=128
```

(4) End2End training pertained transformer + variational GP 				RSE:	0.023642




End2End training random initialed transformer + variational GP.   Dataset size vs RSE
=========================================

```
python train.py -data_train=data_train.json -data_dev=data_test.json -vocab=vocab.json -batch_size=512 -dropout=0.1 -log=./logs/batch_gp.log -save_model=save_model/batch_gp.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=700 -n_layers=1 -d_model=128
```

0.0317

```
python train.py -data_train=data_train_10.json -data_dev=data_test.json -vocab=vocab_10.json -batch_size=256 -dropout=0.1 -log=./logs/split_10.log -save_model=save_model/split_10.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=1100 -n_layers=1 -d_model=128
```

0.199400

```
python train.py -data_train=data_train_50.json -data_dev=data_test.json -vocab=vocab_50.json -batch_size=512 -dropout=0.1 -log=./logs/split_50.log -save_model=save_model/split_50.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=500 -n_layers=1 -d_model=128
```

0.094086

```
python train.py -data_train=data_train_90.json -data_dev=data_test.json -vocab=vocab_90.json -batch_size=512 -dropout=0.1 -log=./logs/split_90.log -save_model=save_model/split_90.pt -bleu_valid_every_n=1 -patience=50 -seed=1 -max_seq_len=16 -n_warmup_steps=700 -n_layers=1 -d_model=128
```

0.042048


| Dataset size  | RSE |
| ------------- | ------------- |
| 100&  | 0.0317    |
| 90%   | 0.042048  |
| 50%   | 0.094086  |
| 10%   | 0.199400  |
