(work in progress)

A simple reproduction of the [Tiny Recursion Model](https://arxiv.org/abs/2510.04871) (TRM) trained on the sudoku-extreme task.

Mainly meant for (my) educational purposes. Tries to follow the notation and pseudocode from the paper, but greatly simplifies the model arch and training.

Also see the [official](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and [@lucidrains'](https://github.com/lucidrains/tiny-recursive-model) implementations.

Works ok. A small model (1.5M) slightly surpasses HRM’s performance (56%) but plateaus at around a 60% solve rate:

```bash
# disabled halting for testing
 uv run main.py --batch_size 128 --h_dim 256 --N_supervision 8 --halt_loss_weight 0.0 --halt_prob_thresh 2.0
```

<img width="541" height="256" alt="1-5M preelim perf" src="https://github.com/user-attachments/assets/a0b2249d-5fed-4b08-a045-e469277b7d2e" />
