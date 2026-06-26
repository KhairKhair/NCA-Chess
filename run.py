import os, sys, json, itertools, time, traceback
import torch
from model import CA, VISIBLE
from dataset import load_split
from train import train, move_accuracy

CACHE = "humanpolicy_10000.pt"
RESULTS = "results_policy"

GRID = {
    "hidden":   [50, 100, 25],
    "update_rate":   [0.5, 1.0, 0.25],
}

def all_configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]

CONFIGS = all_configs(GRID)
print(f"total configs: {len(CONFIGS)}  (use --array=0-{len(CONFIGS) - 1})")

task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
if task >= len(CONFIGS):
    print(f"task {task} >= {len(CONFIGS)}; nothing to do")
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

cfg = CONFIGS[task]
print(f"=== task {task}/{len(CONFIGS) - 1} cfg={cfg} ===", flush=True)

train_data, test_data = load_split(cache=CACHE, test_frac=0.1, seed=0)
Xtr, fltr, tltr, fens_tr = train_data
Xte, flte, tlte, fens_te = test_data

os.makedirs(RESULTS, exist_ok=True)

try:
    model = CA(chn=VISIBLE+12,hidden_n=cfg["hidden"])

    t0 = time.time()
    history = train(
        model, train_data, test_data,
        epochs=200, bs=1024, update_rate=cfg["update_rate"])
    dur = time.time() - t0

    result = {
        "task": task,
        "cfg": cfg,
        "device": str(device),
        "status": "ok",
        "final_train_acc": move_accuracy(model, Xtr[:500], fens_tr[:500],
                                         fltr[:500], tltr[:500], device=device),
        "final_test_acc":  move_accuracy(model, Xte[:500], fens_te[:500],
                                         flte[:500], tlte[:500], device=device),
        "best_test_acc":   max(history["test_acc"]),
        "final_loss":      history["loss"][-1],
        "seconds":         dur,
        "history":         history,
    }

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, f"{RESULTS}/model_{task:03d}.pt")

except Exception as e:
    result = {
        "task": task, "cfg": cfg, "status": "error",
        "error": str(e), "trace": traceback.format_exc(),
    }

with open(f"{RESULTS}/run_{task:03d}.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"task {task} done: best_test={result.get('best_test_acc', 'ERR')}", flush=True)
