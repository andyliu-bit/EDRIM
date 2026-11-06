#!/usr/bin/env bash
# 10-21: B5_seed_44 hit a CUDA/host out-of-memory at 14:00 (free commit 1.3 GB) and the old dispatcher logged it as
# exit=0. Rerun it now as a one-off 4th job (user 10-21, after checking memory: 3 runs use ~6 GB commit each, 14 GB
# free). The dispatcher's stale-claim cleanup keeps the leftovers as *_interrupted_<stamp> and runs the same arm and
# seed from scratch (a rerun of a crash, not a seed swap); its memory gate still applies. The main dispatcher keeps
# MAX_PAR=3, so it starts no new job until the running count drops below 3.
cd C:/LHZ/research/bisurfgnn-model-experiments
MAX_PAR=4 bash experiments/model_improvement/ablation/dispatch_ablation.sh B5_seed_44
