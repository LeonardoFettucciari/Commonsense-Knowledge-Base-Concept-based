#!/bin/bash

python src/utils/extra/upload2gdrive.py \
  --input-dir outputs/upload2gdrive \
  --drive-folder-id 1iJuvppKm8w4Dsk4baHUXZaySNbjZREmp \
  --credentials settings/gen-lang-client-0439062772-a6373d0d2e85.json \
  --delete-original
