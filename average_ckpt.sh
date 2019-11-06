PYTHONPATH=. python scripts/average_checkpoints.py \
    --inputs transformer_attn_500/ \
    --num-epoch-checkpoints 10 --output transformer_attn_500/averaged_model.pt
