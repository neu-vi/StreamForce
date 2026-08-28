# Base model weights

The code loads the Wan2.2 base model from this directory **by path, not by repo id**, so the
layout below has to be exact or you get a `FileNotFoundError` from `utils/wan_wrapper.py`.

```bash
pip install "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir wan_models/Wan2.2-TI2V-5B
```

Expected tree (~33 GB):

```
wan_models/Wan2.2-TI2V-5B/
├── models_t5_umt5-xxl-enc-bf16.pth      # text encoder      utils/wan_wrapper.py:50
├── google/umt5-xxl/                     # tokenizer         utils/wan_wrapper.py:55
├── Wan2.2_VAE.pth                       # VAE               utils/wan_wrapper.py:192
├── config.json                          # DiT
└── diffusion_pytorch_model-*.safetensors
```

A symlink works if the weights already live elsewhere:

```bash
ln -s /path/to/Wan2.2-TI2V-5B wan_models/Wan2.2-TI2V-5B
```

Everything here is gitignored except this file. All scripts resolve `wan_models/` relative to
the current directory, so run them from the repository root.
