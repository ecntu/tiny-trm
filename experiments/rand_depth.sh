#!/bin/bash

# Try random depth idea with n, T and both

mkdir -p models/rand_depth

for seed in 1 2 3; do

    uv run main.py --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 12_500 --test_size 8192 --N_sup_test 256 \
      --checkpoint "models/rand_depth/small_baseline_$seed.pt" \
      --seed "$seed"

    uv run main.py --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 12_500 --test_size 8192 --N_sup_test 256 \
      --init_state random \
      --checkpoint "models/rand_depth/small_only-init_$seed.pt" \
      --seed "$seed"

    uv run main.py --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 12_500 --test_size 8192 --N_sup_test 256 \
      --init_state random --rand_n \
      --checkpoint "models/rand_depth/small_n_$seed.pt" \
      --seed "$seed"

    uv run main.py --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 12_500 --test_size 8192 --N_sup_test 256 \
      --init_state random --rand_T \
      --checkpoint "models/rand_depth/small_T_$seed.pt" \
      --seed "$seed"

      uv run main.py --batch_size 128 --h_dim 256 --N_sup 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0 \
      --steps 12_500 --test_size 8192 --N_sup_test 256 \
      --init_state random --rand_n --rand_T \
      --checkpoint "models/rand_depth/small_n-T_$seed.pt" \
      --seed "$seed"
  done
done
