CUDA_VISIBLE_DEVICES=1 python generate.py $data_dir \
        --path $best_checkpoint_path \
        --max-len-b 110 --min-len 50 --lenpen 3.0 \
        --batch-size 72  --beam 10 --remove-bpe sentencepiece --quiet \
        --no-repeat-ngram-size 3 --answer-output-path $result_path
