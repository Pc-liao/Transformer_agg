CUDA_VISIBLE_DEVICES=1 python generate.py /chome/lpc/cnndm_data/truncate_500/cnndm_bin_joindir \
        --path transformer_attn_500_T/checkpoint_last.pt \
        --max-len-b 110 --min-len 50 --lenpen 2.0 \
        --batch-size 72  --beam 10 --remove-bpe sentencepiece --quiet \
        --no-repeat-ngram-size 3 --answer-output-path transformer_attn_500_T/answer
