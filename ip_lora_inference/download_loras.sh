echo "using HF_API_TOKEN: $HF_API_TOKEN"

huggingface-cli download kfirgold99/Piece-it-Together --repo-type model --include "background_generation/pytorch_lora_weights.safetensors" --local-dir  ./weights/
huggingface-cli download kfirgold99/Piece-it-Together --repo-type model --include "character_sheet/pytorch_lora_weights.safetensors" --local-dir  ./weights/