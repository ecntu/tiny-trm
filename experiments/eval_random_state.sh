#!/bin/bash

# Evaluate init_state=random across seeds and k settings
mkdir -p results/init_random

for seed in 1 2 3; do
    checkpoint_name="small_init=random_k=3_seed=${seed}"

    for k in 1 3 5 7; do
        for k_agg in conf mode; do
        
            echo "=== init_state=random seed=$seed k_passes=$k k_agg=$k_agg ==="

            uv run main.py --batch_size 128 --h_dim 256 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
                --steps 10_000 --test_only --N_sup_test 32 --test_size 8192 \
                --k_passes $k --k_agg $k_agg \
                --init_state random --seed "$seed" \
                --checkpoint "models/init_random/${checkpoint_name}.pt"

            cp "models/init_random/${checkpoint_name}.csv" "results/init_random/${checkpoint_name}_k${k}_${k_agg}.csv"
        done
    done
done
