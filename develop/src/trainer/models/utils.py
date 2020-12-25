import os
import torch
from glob import glob
import torch.nn as nn
from logging import getLogger
import pandas as pd

logger = getLogger("model")


def save_model(model, dir, epoch):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir, f"checkpoint-{epoch}.ckpt"))

    print(f"[+] Model is saved. Epoch: {epoch}")


def load_model(model, dir, load_epoch=None, strict=True, device="cuda"):
    params_to_load = {}
    if device == "cpu":
        params_to_load["map_location"] = torch.device("cpu")

    if not os.path.isdir(dir):
        print("[!] Load is failed")
        return -1

    check_points = glob(os.path.join(dir, "checkpoint-*.ckpt"))
    check_points = sorted(
        check_points,
        key=lambda x: int(
            x.split("/")[-1].replace("checkpoint-", "").replace(".ckpt", "")
        ),
    )

    # skip if there are no checkpoints
    if len(check_points) == 0:
        print("[!] Load is failed")
        return -1

    check_point = check_points[-1]

    # If load_epoch has value
    if load_epoch is not None:

        model.load_state_dict(
            torch.load(
                os.path.join(dir, f"checkpoint-{load_epoch}.ckpt"), **params_to_load
            ),
            strict=strict,
        )
        print("[+] Model is loaded")
        print(f"[+] Epoch: {load_epoch}")
        logger.info(f"[+] Model is loaded | Epoch: {load_epoch}")

        return load_epoch

    model.load_state_dict(torch.load(check_point, **params_to_load), strict=strict)
    last_epoch = int(
        check_point.split("/")[-1].replace("checkpoint-", "").replace(".ckpt", "")
    )

    print("[+] Model is loaded")
    print(f"[+] Epoch: {last_epoch}")
    logger.info(f"[+] Model is loaded | Epoch: {last_epoch}")

    return last_epoch


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias"):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)


def inverse_preprocess_data(data, scaler):
    processed_data = pd.DataFrame(
        scaler.inverse_transform(data), index=data.index, columns=data.columns
    )

    return processed_data
