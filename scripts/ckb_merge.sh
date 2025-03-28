#!/bin/bash


python src/ckb_management/ckb_merge.py \
    --destination_ckb_path "data/ckb/cleaned/full_ckb_old.jsonl" \
    --source_ckb_path "data/ckb/raw/ckb_data=wordnet|model=gemini-2.0-flash.jsonl" \
    --source_model "gemini"