for init_mode in random learned buffer; do
    uv run main.py --batch_size 128  --h_dim 256 \
    --N_supervision 8  --init_state $init_mode --checkpoint_path ./$init_mode.pt --steps 5000 --log_wandb
done
