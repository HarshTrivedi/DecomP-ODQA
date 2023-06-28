#!/bin/bash

set -e
set -x

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

mkdir -p .temp/

# URL: https://drive.google.com/file/d/1tMlfiYUR5iGSBVHCDQt09xTt3byQ-r5_/view?usp=sharing
gdown "1tMlfiYUR5iGSBVHCDQt09xTt3byQ-r5_&confirm=t" -O .temp/processed_data.zip
unzip  -o .temp/processed_data.zip -x "*.DS_Store"

rm -rf .temp/

# The resulting processed_data/ directory should look like:
# ├── 2wikimultihopqa
# │   ├── annotated_only_train.jsonl
# │   ├── dev.jsonl
# │   ├── dev_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# ├── hotpotqa
# │   ├── annotated_only_train.jsonl
# │   ├── dev.jsonl
# │   ├── dev_subsampled.jsonl
# │   ├── test_subsampled.jsonl
# │   └── train.jsonl
# └── musique
#     ├── annotated_only_train.jsonl
#     ├── dev.jsonl
#     ├── dev_subsampled.jsonl
#     ├── test_subsampled.jsonl
#     └── train.jsonl
