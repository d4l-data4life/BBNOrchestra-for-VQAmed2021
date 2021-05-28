#!/usr/bin/env python

import argparse
import logging
import os
import pandas as pd
import subprocess
from tqdm import tqdm 


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _copy(df, sourcepath, targetpath, name):
    logger.info(f"Copy {name} data to target location...")
    for img in tqdm(df['imgid']):
        sourceimgpath = os.path.join(sourcepath, f"{img}.jpg")
        subprocess.call(f"cp {sourceimgpath} {targetpath}", shell=True)


def combine(clef2019path):
    train_path = os.path.join(clef2019path, "ImageClef-2019-VQA-Med-Training")
    val_path = os.path.join(clef2019path, "ImageClef-2019-VQA-Med-Validation")
    train_df = pd.read_csv(os.path.join(train_path, "C4_Abnormality_train.txt"), delimiter="|", names=["imgid", "question", "answer"])
    val_df = pd.read_csv(os.path.join(val_path, "C4_Abnormality_val.txt"), delimiter="|", names=["imgid", "question", "answer"])
    test_df = pd.read_csv(os.path.join(clef2019path, "VQAMed2019_Test_Questions_w_Ref_Answers.txt"), delimiter="|", names=["imgid", "qtype", "question", "answer"])

    train_df_multiclass = train_df.loc[~train_df['answer'].isin(['yes', 'no'])]
    val_df_multiclass = val_df.loc[~val_df['answer'].isin(['yes', 'no'])]
    test_df_multiclass = test_df.loc[(test_df['qtype']=="abnormality") & (~test_df['answer'].isin(['yes', 'no']))]

    combined_path = os.path.join(clef2019path, "combined_abnormality")
    targetimgs_path = os.path.join(combined_path, "images")
    refrence_path = os.path.join(combined_path, "combined_train_val_test.csv" )

    if not os.path.exists(combined_path):
        subprocess.call(f"mkdir {combined_path}", shell=True)
    if not os.path.exists(targetimgs_path):
        subprocess.call(f"mkdir {targetimgs_path}", shell=True)

    _copy(train_df_multiclass, os.path.join(train_path, "Train_images"), targetimgs_path, "train")
    _copy(val_df_multiclass, os.path.join(val_path, "Val_images"), targetimgs_path, "val")
    _copy(test_df_multiclass, os.path.join(clef2019path, "VQAMed2019_Test_Images"), targetimgs_path, "test")

    df_combined = train_df_multiclass.append(val_df_multiclass)
    df_combined = df_combined.append(test_df_multiclass)
    df_combined.drop(columns=["qtype"], inplace=True)
    df_combined.to_csv(refrence_path, sep="|", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare clef2019 for training. Create a new directory and copies combined train, validation and test sets deom VQA Med 2019 abnormality question type.")
    parser.add_argument(type=str, dest='clef2019path', help='Path to the directory containing train, validation and test set from VQA-Med 2019.')

    clef2019path = parser.parse_args().clef2019path

    combine(clef2019path)

