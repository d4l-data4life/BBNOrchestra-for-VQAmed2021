#!/usr/bin/env python

import _init_paths
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from utils.utils import create_logger
from net import Network
from tqdm import tqdm
import numpy as np
import os
import json
from torchvision import models
from config import cfg, update_config
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm


def _most_frequent(List):
    return max(set(List), key = List.count)


def _predict(dataloader, model, cfg, num_classes=330, mode='valid'):
    pbar = tqdm(total=(len(dataloader)))
    model.eval()
    func = torch.nn.Softmax(dim=1)
    probs_list = []
    labels_list = []  # predicted label index
    gtlabel_names = []  # ground truth
    image_names = []
    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataloader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            probs, labels = result.topk(1, 1, True, True)
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                probs_list.append(probs[i])
                labels_list.append(labels[i])
                if mode == "valid":
                    gtlabel_names.append(meta["actual_label"][i])
                image_names.append(meta["image_name"][i])
            pbar.update(1)
    return probs_list, labels_list, gtlabel_names, image_names


def voted_predictions(cfg, logger, device):
    # make predictions
    with open(cfg.DATASET.CATID_LABEL_JSON) as f:
        catid_to_label = json.load(f)
    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    members = []

    models_path_root = cfg.TEST.MODEL_FILE
    logger.info("Loading models...")
    for k in range(cfg.DATASET.N_SPLITS):
        model_path = os.path.join(models_path_root, f"models_{k}split/best_model.pth")
        model = Network(cfg, mode="test", num_classes=num_classes)
        model.load_model(model_path)
        model = model.to(device)
        model.eval()
        members.append(model)

    testloader = DataLoader(
            test_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            )
    data_size = len(testloader.dataset)
    sum_probs = [0 for i in range (data_size)]
    best_labels = [0 for i in range (data_size)]
    result_list = [dict() for i in range (data_size)]
    logger.info("Choosing best prediction...")
    votes = [[] for i in range(data_size)]
    for model in tqdm(members):
        model = model.to(device)
        pred_probs, pred_labels, gtlabelnames, image_names = _predict(testloader, model, cfg)
        for i in range(data_size):
            votes[i].append(pred_labels[i])
    most_voted = [_most_frequent(preds) for preds in votes]
    for i in range(data_size):
        result_list[i] = {
                "image_name": image_names[i],
                "groundtruth_label": gtlabelnames[i],
                "predicted_label": catid_to_label[str(most_voted[i].item())]
                }

    assert len(result_list) == data_size
    acc = 0 
    for res in result_list:
        if res["groundtruth_label"] == res["predicted_label"]:
            acc += 1

    print(acc)
    print(data_size)
    acc = acc * 100 / data_size
    print(f"Top1 accuracy: {acc}")

    return result_list 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBN evaluation")
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=True,
            default="../configs/BBN-ResNeSt-orchestra.yaml",
            type=str,
            )
    parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
            )
    parser.add_argument(
            "--gpu",
            help="gpu instance index",
            required=False,
            default=0,
            type=int,
            )

    args = parser.parse_args()
    update_config(cfg, args)
    logger, log_file = create_logger(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    voted_predictions(cfg, logger, device)
