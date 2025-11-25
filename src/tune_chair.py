import json
import subprocess
from pathlib import Path

# Path to your features CSV (same as you pass to train_chair_nn.py)
FEATURES = "chair_data/chair_features.csv"

# Where to save the best hyperparameters
KNOBS_PATH = Path("outputs/chair_nn_knobs.json")

def main():
    # --- Hyperparameter grid ---
    lrs = [1e-5, 3e-5, 1e-4]
    d_models = [64, 128]
    num_layers_list = [1, 2]
    nheads = [2, 4]       # must divide d_model
    batch_sizes = [32]
    epochs_list = [10]    # you can add [5, 10, 15] if you want

    best_auc = -1.0
    best_cfg = None

    for lr in lrs:
        for d_model in d_models:
            for num_layers in num_layers_list:
                for nhead in nheads:
                    if d_model % nhead != 0:
                        continue  # skip invalid combos

                    for batch_size in batch_sizes:
                        for epochs in epochs_list:
                            out_name = (
                                f"outputs/tune_tests/nn_dm{d_model}_L{num_layers}_h{nhead}"
                                f"_lr{lr}_bs{batch_size}_ep{epochs}.pth"
                            )

                            cmd = [
                                "python",
                                "src/train_chair_nn.py",
                                "--features", FEATURES,
                                "--out", out_name,
                                "--lr", str(lr),
                                "--d_model", str(d_model),
                                "--nhead", str(nhead),
                                "--num_layers", str(num_layers),
                                "--batch_size", str(batch_size),
                                "--epochs", str(epochs),
                            ]

                            print("\n============================")
                            print("Running:", " ".join(cmd))
                            print("============================\n")
                            subprocess.run(cmd, check=True)

                            metrics_path = Path(out_name).with_suffix(".metrics.json")
                            if not metrics_path.exists():
                                print(f"WARNING: metrics file not found: {metrics_path}")
                                continue

                            with open(metrics_path) as f:
                                metrics = json.load(f)
                            val_auc = metrics["val"]["auc_roc"]

                            print(f"  -> val AUC for this run: {val_auc:.4f}")

                            if val_auc > best_auc:
                                best_auc = val_auc
                                best_cfg = {
                                    "lr": lr,
                                    "d_model": d_model,
                                    "num_layers": num_layers,
                                    "nhead": nhead,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "out": str(out_name),
                                }
                                print("\n*** NEW BEST ***")
                                print("AUC:", best_auc)
                                print("CFG:", best_cfg)
                                print("****************\n")

    print("\n===== GRID SEARCH DONE =====")
    print("BEST AUC:", best_auc)
    print("BEST CONFIG:", best_cfg)

    if best_cfg is None:
        print("No successful runs; nothing to save.")
        return

    # --- Save knobs JSON with best config ---
    knobs_payload = {
        "best_val_auc": best_auc,
        "config": best_cfg,
    }
    KNOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KNOBS_PATH, "w") as f:
        json.dump(knobs_payload, f, indent=2)
    print(f"Saved best hyperparameters to {KNOBS_PATH}")

    # --- Re-train once with best config, writing to default out path ---
    final_out = "outputs/chair_classifier_nn.pth"  # default in train_chair_nn.py

    final_cmd = [
        "python",
        "src/train_chair_nn.py",
        "--features", FEATURES,
        "--out", final_out,
        "--lr", str(best_cfg["lr"]),
        "--d_model", str(best_cfg["d_model"]),
        "--nhead", str(best_cfg["nhead"]),
        "--num_layers", str(best_cfg["num_layers"]),
        "--batch_size", str(best_cfg["batch_size"]),
        "--epochs", str(best_cfg["epochs"]),
    ]

    print("\nRe-training best config to default output:")
    print(" ".join(final_cmd))
    subprocess.run(final_cmd, check=True)
    print(f"Final best model written to {final_out}")

if __name__ == "__main__":
    main()