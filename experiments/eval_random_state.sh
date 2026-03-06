#!/bin/bash

# Compare init_state=random vs init_state=buffer across 3 seeds
mkdir -p results/eval_random

# small models (without halting)
mode=random
for seed in 1 2 3 4 5; do

    checkpoint_name="small_init_${mode}_seed${seed}"

    for k in 1 3 5 7; do

        echo "=== init_state=$mode seed=$seed k_passes=$k ==="

        uv run main.py --batch_size 128 --h_dim 256 --N_supervision_test 256 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
        --steps 10_000 --test_only --test_size 12800 --k_passes $k \
        --init_state "random" \
        --checkpoint "models/${checkpoint_name}.pt"
        
        cp "models/${checkpoint_name}.csv" "results/eval_random/${checkpoint_name}_${k}.csv"
    done 
done
