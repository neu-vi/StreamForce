#!/usr/bin/env bash
# Rolling forcing + SERVER-SIDE GENERATION PACING (experiment).
#
# Same as this demo, plus: the generator holds at a block boundary while it
# is more than PACE_HIGH video frames ahead of the frame the browser reports showing, and resumes
# once the gap falls to PACE_LOW. Nothing is dropped and nothing is regenerated -- it only stops
# the generator racing ahead of the viewer, so a force change lands near what you are watching.
#   PACE_HIGH=48   hold above this gap (0 disables pacing = original behaviour)
#   PACE_LOW=24    resume at or below this gap
#
# Identical to this demo in every respect except that generation runs through
# pipeline/rolling_forcing_streaming_inference.py (rolling denoising window + attention sink)
# instead of the causal self-forcing pipeline. Same UI, transport, save/arrows and self-test.
#
#   ./demo/run.sh
#
# Runs from the repo root with the streamforce env active. Forward the port when remote:
#   ssh -L 5013:localhost:5013 <host>
#
# One checkpoint, used for both point and wind. Pass it in:
#   CHECKPOINT=<PATH_TO_DISTILLED_STUDENT_CKPT> ./demo/run.sh
# Add DEVICE=cuda:1 to move the model off GPU 0.
#
# Wire knobs (all optional; generation is unaffected by every one of them):
#   JPEG_Q=60        JPEG quality for frames sent to the browser (default 85)
#   WIRE_SCALE=0.75  downscale only what is sent (default 1.0 = off)
#
# Bind address: use DEMO_HOST, not HOST -- conda's compiler activation exports
# HOST=x86_64-conda-linux-gnu, which werkzeug would try to resolve and then exit with
# "Name or service not known".
#   DEMO_HOST=0.0.0.0  to accept connections from outside the box
#
# Performance: every optimization is ON by default. See OPTIMIZATIONS.md for what each one
# needs (extra GPU / package) and what it costs numerically. To turn one off:
#   VAE_DEVICE=same        keep the VAE on the generator's GPU (single-GPU mode; auto-detected)
#   VAE_DEVICE=cuda:3      pin the VAE to a specific GPU
#   VAE_CHANNELS_LAST=0    bit-identical VAE decode, ~3.6x slower
#   GEN_CHANNELS_LAST=0    bit-identical generator patch-embed
#   WAN_ASYNC_VAE=0        decode inline instead of on a worker thread
#
# Rolling-forcing knobs:
#   RF_BLOCK=3       latent frames per block == attention-sink size
#   RF_WINDOW=21     latent frames the attention window reads back over
#   LATENTS=126      raise it: rolling forcing evicts, so runs are not capped at 126
set -euo pipefail

cd "$(dirname "$0")/.."

# The site-wide HF_HOME may point at a shared cache that is read-only for us and does not
# hold the captioner weights. Fall back to the per-user cache when we cannot write to it.
if [ -n "${HF_HOME:-}" ] && [ ! -w "${HF_HOME}/hub" ]; then
    export HF_HOME="$HOME/.cache/huggingface"
    echo "[run.sh] HF_HOME was not writable; using $HF_HOME"
fi

# The model takes one GPU; the rest is headroom for the captioner's device_map=auto.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python demo/app.py \
    --config_path        "${CONFIG:-configs/dmd_everything.yaml}" \
    --checkpoint         "${CHECKPOINT:?set CHECKPOINT=/path/to/model.pt}" \
    --output_dir         "${OUTPUT_DIR:-demo/demo_outputs}" \
    --height             "${HEIGHT:-480}" \
    --width              "${WIDTH:-832}" \
    --num_latent_frames  "${LATENTS:-126}" \
    --rolling_forcing_block_frames "${RF_BLOCK:-3}" \
    --rolling_forcing_max_frames   "${RF_WINDOW:-21}" \
    --seed               "${SEED:-0}" \
    --host               "${DEMO_HOST:-127.0.0.1}" \
    --port               "${PORT:-5013}" \
    --caption_model      "${CAPTION_MODEL-Qwen/Qwen3-VL-8B-Instruct}" \
    --caption_device     "${CAPTION_DEVICE:-auto}" \
    --pace_high_frames   "${PACE_HIGH:-45}" \
    --pace_low_frames    "${PACE_LOW:-40}" \
    --pace_min_buffer    "${PACE_MIN_BUFFER:-6}" \
    --pace_buffer_margin "${PACE_BUFFER_MARGIN:--1}" \
    --jpeg_quality       "${JPEG_Q:-85}" \
    --wire_scale         "${WIRE_SCALE:-1.0}" \
    ${DEVICE:+--device "$DEVICE"} \
    --vae_device         "${VAE_DEVICE:-auto}" \
    ${VAE_CHANNELS_LAST:+$([ "$VAE_CHANNELS_LAST" = 0 ] && echo --no_vae_channels_last)} \
    ${GEN_CHANNELS_LAST:+$([ "$GEN_CHANNELS_LAST" = 0 ] && echo --no_gen_channels_last)}
