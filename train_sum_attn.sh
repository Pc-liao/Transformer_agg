CUDA_VISIBLE_DEVICES=1 python train.py \
    /yourdata_dir \
    --arch transformer_agg_copy --share-decoder-input-output-embed \
    --share-all-embeddings \
    --agg-method attn --agg-layers 1 \
     --lr-shrink 0.5 --update-freq 8 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
    --lr 0.0001 --lr-scheduler reduce_lr_on_plateau  \
    --dropout 0.1 --num-workers 8 \
    --encoder-layers 4 --decoder-layers 4 \
    --criterion cross_entropy \
    --max-tokens 6666 --save-dir transformer_agg_copy/  \
    --skip-invalid-size-inputs-valid-test 
