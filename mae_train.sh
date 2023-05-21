python main_pretrain.py \
    --nodes 8 \
    --batch_size 64 \
    --model mae_deit_tiny_patch4 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path './data'