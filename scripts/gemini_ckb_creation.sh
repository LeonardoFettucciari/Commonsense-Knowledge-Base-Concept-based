# Store the API key
API_KEY="$1"

python src/ckb_creation/gemini_ckb_creation.py \
    --config_path settings/gemini_config.json \
    --api_key "$API_KEY" \
    --output_dir outputs/ckb \
