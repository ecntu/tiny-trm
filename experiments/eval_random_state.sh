#!/bin/bash

# Compare init_state=random vs init_state=buffer across seeds and k settings
mkdir -p results/eval_random

for mode in random buffer; do
    # Match k_passes used during training (from init_state.sh)
    train_k=$( [ "$mode" = "random" ] && echo 3 || echo 1 )

    for seed in 1 2 3; do
        checkpoint_name="small_init=${mode}_k=${train_k}_seed=${seed}"

        for k in 1 3 5 7; do
            for k_mode in conf mode; do

                # skip redundant k_mode when k=1
                if [ "$k" -eq 1 ] && [ "$k_mode" = "mode" ]; then continue; fi

                echo "=== init_state=$mode seed=$seed k_passes=$k k_mode=$k_mode ==="

                uv run main.py --batch_size 128 --h_dim 256 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
                    --steps 10_000 --test_only --N_sup_test 32 --test_size 8192 \
                    --k_passes $k --k_mode $k_mode \
                    --init_state "$mode" --seed "$seed" \
                    --checkpoint "models/${checkpoint_name}.pt"

                cp "models/${checkpoint_name}.csv" "results/eval_random/${checkpoint_name}_k${k}_${k_mode}.csv"
            done
        done
    done
done
