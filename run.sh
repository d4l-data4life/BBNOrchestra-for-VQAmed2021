#!/bin/bash

# Inputs for prep 2019 data
CLEF2019PATH=/path/to/imageCLEF2019
PREPPROG=./prep_clef2019.py

# Inputs for create json
TRAIN2020PATH=/path/to/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet
VAL2020PATH=/path/to/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet
COMBINED2019PATH=/path/to/imageCLEF2019/combined_abnormality/
VAL2021PATH=/path/to/VQA-Med-2021-Tasks-1-2-NewValidationSets
TEST2021PATH=/path/to/VQA-Med-2021-TestSets/VQA-2021-TestSet
OUTJSONPATH=./jsons
CREATEPROG=./create_jsons.py

# Inputs for BBN-Orchestra train
CONFIGFILE=./configs/BBN-ResNeSt-orchestra.yaml
TRAINPROG=./main/train_orchestra.py
VALIDPROG=./main/valid_orchestra.py

# Pipeline
chmod +x $PREPPROG
chmod +x $CREATEPROG
chmod +x $TRAINPROG
chmod +x $VALIDPROG

echo "Preparing VQA-Med 2019 data..."
$PREPPROG $CLEF2019PATH

echo "Creating json files for BBN..."
$CREATEPROG $TRAIN2020PATH $VAL2020PATH $COMBINED2019PATH $VAL2021PATH $TEST2021PATH $OUTJSONPATH

echo "Training BBN-Orchestra..."
$TRAINPROG --cfg $CONFIGFILE
echo "Validating BBN-Orchestra..."
$VALIDPROG --cfg $CONFIGFILE
