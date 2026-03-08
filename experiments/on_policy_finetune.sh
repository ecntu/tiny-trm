#!/bin/bash

# Finetune small buffer-init models from the init_random checkpoints.
mkdir -p models/on_policy

mode=buffer
k_passes=1
checkpoint_dir="models/init_random"

for seed in 1 2 3; do
    checkpoint_name="small_init=${mode}_k=${k_passes}_seed=${seed}.pt"
    source_checkpoint="${checkpoint_dir}/${checkpoint_name}"

    for variant in off_policy on_policy; do
        new_checkpoint="models/on_policy/${variant}_${checkpoint_name}"

        cp "$source_checkpoint" "$new_checkpoint"

        extra_args=()
        if [ "$variant" = "on_policy" ]; then
            extra_args+=(--stay_on_policy)
        fi

        # First line of args must match how og checkpoint was trained
        uv run main.py \
            --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 --init_state "$mode" --k_passes "$k_passes" \
            --steps 3_000 --lr 3e-5 --checkpoint "$new_checkpoint" --seed "$seed" \
            --test_size 8192 --N_sup_test 256 \
            "${extra_args[@]}"
    done
done