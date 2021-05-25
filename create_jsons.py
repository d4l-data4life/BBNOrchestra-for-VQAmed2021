#!/usr/bin/env python

import argparse
import json
import logging
import os
import pandas as pd
from tqdm import tqdm


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _update_annotations(df, basepath, labels_to_ids, annotations, name, test=False):
    logger.info(f"Update annotations with {name}...")
    for i, row in tqdm(df.iterrows()):
        imgpath = os.path.join(basepath, f"{row['imgid']}.jpg")
        if not test:
            if row['answer'] not in labels_to_ids:
                continue   # skip the non-intersecting answers from clef 2019
            innerdict = {
                    "category_id": labels_to_ids[row['answer']],
                    "image_id": row['imgid'],
                    "fpath": imgpath,
                    "image_label": row['answer']
                    }
        else:
            innerdict = {
                    "category_id": "",
                    "image_id": row['imgid'],
                    "fpath": imgpath,
                    "image_label": ""
                    }

        annotations.append(innerdict)

    return annotations


def create_jsons(train2020path, val2020path, clef2019path, val2021path, test2021path, jsonpath):
    # read date
    train2020_df = pd.read_csv(os.path.join(train2020path, "VQAnswering_2020_Train_QA_pairs.txt"), delimiter='|', names=['imgid', 'question', 'answer'])
    val2020_df = pd.read_csv(os.path.join(val2020path, "VQAnswering_2020_Val_QA_Pairs.txt"), delimiter='|', names=['imgid', 'question', 'answer'])
    clef2019_df = pd.read_csv(os.path.join(clef2019path, "combined_train_val_test.csv"), delimiter='|')
    val2021_df = pd.read_csv(os.path.join(val2021path, "VQA-Med-2021-VQAnswering-Task1-New-ValidationSet.txt"), delimiter='|', names=['imgid', 'question', 'answer'])
    test2021_df = pd.read_csv(os.path.join(test2021path, "Task1-VQA-2021-TestSet-Questions.txt"), delimiter='|', names=['imgid', 'question'])
    
    # omit yes/no data
    train2020_multiclass = train2020_df.loc[~train2020_df['answer'].isin(["yes", "no"])]
    val2020_multiclass = val2020_df.loc[~val2020_df['answer'].isin(["yes", "no"])]
    clef2019_multiclass = clef2019_df.loc[~clef2019_df['answer'].isin(["yes", "no"])]

    # create data dictionary for BBN
    logger.info("Creating train json for BBN ensemble training...")
    annotations = []
    labels = train2020_multiclass['answer'].unique()
    labels_to_ids = {k: v for v, k in enumerate(labels)}
    annotations = _update_annotations(train2020_multiclass, os.path.join(train2020path, "VQAnswering_2020_Train_images"), labels_to_ids, annotations, "train2020")
    annotations = _update_annotations(val2020_multiclass, os.path.join(val2020path, "VQAnswering_2020_Val_images"), labels_to_ids, annotations, "val2020")
    annotations = _update_annotations(clef2019_multiclass, os.path.join(clef2019path, "images"), labels_to_ids, annotations, "clef2019")
    annotations = _update_annotations(val2021_df, os.path.join(val2021path, "ImageCLEF-2021-VQA-Med-New-Validation-Images"), labels_to_ids, annotations, "val2021")

    bbn_dict = {
            "num_classes": len(labels),
            "annotations": annotations
            }


    logger.info("Creating test json for BBN...")
    test_annotations = []
    test_annotations = _update_annotations(test2021_df, os.path.join(test2021path, "Task1-VQA-2021-TestSet-Images"), {}, test_annotations, "test 2021", test=True)

    test_bbn_dict = {
            "num_classes": len(labels),
            "annotations": test_annotations
            }

    # dump dictionary to json
    with open(os.path.join(jsonpath, "train.json"), 'w') as f:
        json.dump(bbn_dict, f)
    with open(os.path.join(jsonpath, "test.json"), 'w') as f:
        json.dump(test_bbn_dict, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create input json using train and validation sets for BBN Orchestra.")
    parser.add_argument(type=str, dest='train2020path', help='Path to the train2020 directory containing images folder and txt file.')
    parser.add_argument(type=str, dest='val2020path', help='Path to the val2020 directory containing images and txt file.')
    parser.add_argument(type=str, dest='clef2019path', help='Path to the clef2019 directory containing images and txt file.')
    parser.add_argument(type=str, dest='val2021path', help='Path to the val2021 directory containing images and txt file.')
    parser.add_argument(type=str, dest='test2021path', help='Path to the test2021 directory containing images and txt file.')
    parser.add_argument(type=str, dest='jsonpath', help='Path to the directpry to dump data the output jsons.')

    train2020path = parser.parse_args().train2020path
    val2020path = parser.parse_args().val2020path
    clef2019path = parser.parse_args().clef2019path
    val2021path = parser.parse_args().val2021path
    test2021path = parser.parse_args().test2021path
    jsonpath = parser.parse_args().jsonpath

    create_jsons(train2020path, val2020path, clef2019path, val2021path, test2021path, jsonpath)
