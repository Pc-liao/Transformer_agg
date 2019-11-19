# Transformer_agg

data could be download: https://drive.google.com/drive/folders/1SlZ9PFlEIizh5r5vcmCLzkaZI34p9P3T?usp=sharing

## truncate and use bpe
1.move `scripts/truncate_data.py` to your dataset dir and modify the truncate length then run 
``` commandline
python truncate_data.py
```
2.modify data dir as the location of your data in `prepro.py` then run
```commandline
python prepro.py
```
## preprocessing
You also need modify `$raw_data_dir` and `$data_dir` in script `preprocess.sh`
``` commandline
chmod u+x preprocess.sh # optional, if you dont have permission
bash preprocess.sh
```

## training
modify `$data_dir` and `$model_saved_dir` in `train_sum_attn.sh` 
``` commandline
chmod u+x train_sum_attn.sh # optional, if you dont have permission
bash train_sum_attn.sh
```

## generating
modify `$data_dir` and `$result` in `generate_cnn.sh` 
``` commandline
chmod u+x generate_cnn.sh # optional, if you dont have permission
bash generate_cnn.sh
```

## Reference:
 fairseq: https://github.com/pytorch/fairseq
