# SAMP Training Guide (JupyterHub)

Shape-Aware Manifold Packing — CS 652 Project  
Run all experiments from the repo root: `~/clamp/` (or wherever you cloned the repo).

---

## 1. One-Time Setup

### 1.1 Install dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Verify GPU is visible
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
Expected: something like `NVIDIA A100-SXM4-40GB MIG 3g.20gb`

---

## 2. Use tmux (Prevents Session Timeouts)

Always run training inside a tmux session so it keeps going if your browser disconnects.

```bash
# Start a new session
tmux new -s train

# Detach (leave running in background)
# Press:  Ctrl+B, then D

# Reattach later
tmux attach -t train

# List all sessions
tmux ls
```

To run multiple experiments in parallel, create one window per run:
```bash
# Inside tmux: open a new window
# Press:  Ctrl+B, then C

# Switch between windows
# Press:  Ctrl+B, then 0  (or 1, 2, 3...)

# Rename current window (optional)
# Press:  Ctrl+B, then ,
```

---

## 3. Environment Variables (Shared Memory Fix)

If you see `Bus error` or workers crashing, the JupyterHub `/dev/shm` is too small.
The configs already set `cpus_per_gpu = 2` which helps, but if it still crashes:

```bash
export CLAMP_FORCE_SINGLE_WORKER=1
```

Run this before any training command in that tmux window.

---

## 4. Training Commands

All commands follow this pattern:
```bash
python pretrain.py <examples/run-folder> <default_configs/default_config_cifar10.ini>
```

### Run 1 — CIFAR-100 Baseline (original CLAMP loss)
```bash
python pretrain.py examples/cifar100-baseline default_configs/default_config_cifar10.ini
```

### Run 2 — Anisotropic Overlap Detection only
```bash
python pretrain.py examples/cifar100-anisotropic default_configs/default_config_cifar10.ini
```

### Run 3 — Full SAMP, γ = 0.0
```bash
python pretrain.py examples/cifar100-samp-g0 default_configs/default_config_cifar10.ini
```

### Run 4 — Full SAMP, γ = 0.25
```bash
python pretrain.py examples/cifar100-samp-g025 default_configs/default_config_cifar10.ini
```

### Run 5 — Full SAMP, γ = 0.50
```bash
python pretrain.py examples/cifar100-samp-g050 default_configs/default_config_cifar10.ini
```

### Run 6 — Full SAMP, γ = 0.75
```bash
python pretrain.py examples/cifar100-samp-g075 default_configs/default_config_cifar10.ini
```

### Run 7 — Full SAMP, γ = 1.0 (equivalent to anisotropic with no gradient attenuation)
```bash
python pretrain.py examples/cifar100-samp-g100 default_configs/default_config_cifar10.ini
```

---

## 5. Linear Evaluation (after pretraining)

Run after pretraining completes for each experiment:

```bash
python linear_probe.py examples/cifar100-baseline default_configs/default_config_cifar10.ini
default_configs/default_config_cifar10.ini
python linear_probe.py examples/cifar100-samp-g0 default_configs/default_config_cifar10.ini
python linear_probe.py examples/cifar100-samp-g025 default_configs/default_config_cifar10.ini
python linear_probe.py examples/cifar100-samp-g050 default_configs/default_config_cifar10.ini
python linear_probe.py examples/cifar100-samp-g075 default_configs/default_config_cifar10.ini
python linear_probe.py examples/cifar100-samp-g100 default_configs/default_config_cifar10.ini
```

---

## 6. Recommended Run Order

Run 1 first as a sanity check. Once epoch 1 completes without error, launch the rest in separate tmux windows.

```
Window 0:  Run 1  (baseline)
Window 1:  Run 2  (anisotropic)
Window 2:  Run 3  (samp g0)
Window 3:  Run 4  (samp g025)
Window 4:  Run 5  (samp g050)
Window 5:  Run 6  (samp g075)
Window 6:  Run 7  (samp g100)
```

> Note: All 7 runs simultaneously may saturate the GPU. Start 2-3 at a time and monitor memory.

---

## 7. Monitor Training Progress

### Check GPU memory usage
```bash
nvidia-smi
```

### Watch training metrics in real time
```bash
# Replace with whichever run you want to watch
tail -f examples/cifar100-baseline/ssl/metrics.csv
```

### View TensorBoard logs
```bash
tensorboard --logdir examples/ --port 6006
```
Then open `http://localhost:6006` in your browser.

### Key metrics to watch
| Metric | Healthy trend |
|--------|--------------|
| `train_loss` | Decreasing (log scale) |
| `val_acc` | Increasing toward ~55-65% by epoch 1000 |
| `mean_radius` | Decreasing then stabilizing |
| `num_nbr` | Decreasing toward 0 (ellipsoids separating) |
| `mean_dist` | Increasing (ellipsoids spreading out) |

---

## 8. Smoke Test (Quick Sanity Check)

Before committing to 1000-epoch runs, verify the new loss classes work by running 2 epochs:

1. Temporarily edit a config:
   ```
   n_epochs = 2
   save_every_n_epochs = 1
   ```
2. Run it:
   ```bash
   python pretrain.py examples/cifar100-anisotropic default_configs/default_config_cifar10.ini
   ```
3. Confirm: no crash, loss decreases from step 1 to step 2
4. Restore `n_epochs = 1000` and `save_every_n_epochs = 100` before the real run

---

## 9. Checkpoints and Resume

Checkpoints are saved automatically every 100 epochs to `examples/<run>/ssl/`.

If training crashes, just re-run the same command — it will auto-resume from the latest checkpoint.

```bash
# Check what checkpoints exist
ls examples/cifar100-baseline/ssl/
```

---

## 10. Pushing Results to GitHub

After training completes (or at checkpoints):

```bash
git add examples/cifar100-baseline/ssl/metrics.csv
git add examples/cifar100-anisotropic/ssl/metrics.csv
# (add other metrics.csv files as needed — avoid committing .ckpt files, they're in .gitignore)

git commit -m "Add training metrics for baseline and anisotropic runs"
git push origin main
```

> Checkpoint `.ckpt` files are large — do not commit them unless necessary.
