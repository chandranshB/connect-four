"""Export trained PyTorch Connect4Net checkpoints to self-contained ONNX models and verify parity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from c4_env import encode_obs_batch, legal_actions, play_move
from model import Connect4Net, count_parameters


def export_checkpoint(ckpt_path: Path, onnx_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Connect4Net(channels=ckpt["channels"], num_blocks=ckpt["num_blocks"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros((1, 3, 6, 7), dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export using legacy TorchScript exporter / single-file weights so .onnx has zero external data files
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Validate ONNX model graph
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Verify numerical parity across random + empty board states (batch=1 and batch=16)
    rng = np.random.default_rng(999)
    test_states = [(0, 0, 0)]
    for _ in range(15):
        pos, mask, ply = 0, 0, 0
        for _ in range(int(rng.integers(1, 22))):
            legal = legal_actions(mask)
            if not legal:
                break
            pos, mask, ply = play_move(pos, mask, ply, int(rng.choice(legal)))
        test_states.append((pos, mask, ply))

    obs = encode_obs_batch(test_states)
    with torch.inference_mode():
        pt_logits, pt_val = model(torch.from_numpy(obs))
        pt_logits_np = pt_logits.numpy()
        pt_val_np = pt_val.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits, ort_val = sess.run(["policy_logits", "value"], {"board": obs})

    max_logits_diff = float(np.max(np.abs(pt_logits_np - ort_logits)))
    max_val_diff = float(np.max(np.abs(pt_val_np - ort_val)))
    assert max_logits_diff < 1e-4, f"Logits parity mismatch: {max_logits_diff}"
    assert max_val_diff < 1e-4, f"Value parity mismatch: {max_val_diff}"

    # Also test batch=1 specifically
    ort_l1, ort_v1 = sess.run(["policy_logits", "value"], {"board": obs[:1]})
    assert ort_l1.shape == (1, 7) and ort_v1.shape == (1,)

    size_kb = round(onnx_path.stat().st_size / 1024.0, 1)
    return {
        "onnx_file": onnx_path.name,
        "iteration": ckpt["iteration"],
        "params": count_parameters(model),
        "size_kb": size_kb,
        "max_logits_diff": max_logits_diff,
        "max_value_diff": max_val_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Connect4 checkpoints to ONNX for browser")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("../web/public/models"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for tier in ("early", "mid", "pro"):
        ckpt = args.ckpt_dir / f"{tier}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        info = export_checkpoint(ckpt, args.out_dir / f"{tier}.onnx")
        exported[tier] = info
        print(
            f"Exported {tier}.onnx ({info['size_kb']} KB, iter {info['iteration']}, "
            f"max diff={info['max_logits_diff']:.2e})"
        )

    # Copy training report for UI model info readout
    report_src = args.ckpt_dir / "training_report.json"
    if report_src.exists():
        report = json.loads(report_src.read_text(encoding="utf-8"))
        report["exported"] = exported
        (args.out_dir / "model_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        (args.out_dir / "model_info.json").write_text(json.dumps({"exported": exported}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
