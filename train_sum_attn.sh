CUDA_VISIBLE_DEVICES=1 python train.py \
    $data_dir \
    --arch transformer_agg_copy --share-decoder-input-output-embed \
    --share-all-embeddings \
    --agg-method attn --agg-layers 1 \
     --lr-shrink 0.5 --update-freq 8 \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 0.1 \
    --lr 0.0001 --lr-scheduler reduce_lr_on_plateau  \
    --dropout 0.1 --num-workers 8 \
    --encoder-layers 4 --decoder-layers 4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
    --max-tokens 6666 --save-dir $model_saved_dir  \
    --skip-invalid-size-inputs-valid-test 
