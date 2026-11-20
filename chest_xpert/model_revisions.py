"""Hugging Face model revisions used by the application runtime."""

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_CONFIG_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
# The current CLIP revision provides a PyTorch .bin file. Use this earlier
# safetensors revision so startup never relies on loading a pickle checkpoint.
CLIP_SAFETENSORS_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"
