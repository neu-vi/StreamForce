"""Incremental VAE decoding for streaming inference.

Used by `rolling_forcing_streaming_inference.py` to turn latents into frames as they are
generated, instead of decoding the whole clip at the end.
"""
from typing import Callable, List, Optional
import os
import queue
import threading
import time

import torch

from utils.wan_wrapper import WanVAEWrapper
from wan.modules.vae2_2 import unpatchify


class _StreamingVAEChunkDecoder:
    """Incremental decoder for Wan2.2 VAE.

    The stock decode path clears cache per call. For streaming decode, we keep
    decoder cache alive across latent chunks so each new latent contributes only
    new frames (first latent -> 1 frame, later latents -> 4 frames).
    """

    def __init__(self, vae_wrapper: WanVAEWrapper):
        self.vae = vae_wrapper
        self.model = vae_wrapper.model
        self.started = False
        self.model.clear_cache()

    def clear(self) -> None:
        self.started = False
        self.model.clear_cache()

    def decode_latent_chunk(
        self,
        latent_chunk: torch.Tensor,
        on_partial: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Decode a chunk; with `on_partial`, hand over each latent's frames as they are ready.

        The loop below is already per-latent -- Wan's causal VAE carries `feat_cache` across
        frames, so they can only be decoded in order anyway. Waiting for the whole chunk before
        returning was purely a delivery choice, and an expensive one downstream: it made the
        viewer receive one 12-frame burst per block instead of three 4-frame ones, and put a
        whole block's decode (~222 ms) into the refill latency instead of one latent's (~74 ms).

        `on_partial` receives [B, t, C, H, W] in the same units as the return value.
        `unpatchify` is spatial only (the frame axis passes through), so doing it per latent is
        exactly equivalent to doing it on the concatenated tensor.
        """
        # latent_chunk: [B, T, C, H, W]
        if latent_chunk.shape[0] != 1:
            raise ValueError("Streaming decoder currently supports batch_size=1")

        zs = latent_chunk.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        outputs = []
        for u in zs:
            z = u.unsqueeze(0)
            device, dtype = z.device, z.dtype
            mean = self.vae.mean.to(device=device, dtype=dtype)
            inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype)
            z = z / inv_std.view(1, self.model.z_dim, 1, 1, 1) + mean.view(1, self.model.z_dim, 1, 1, 1)

            x = self.model.conv2(z)
            out_list = []
            for i in range(x.shape[2]):
                self.model._conv_idx = [0]
                out_i = self.model.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self.model._feat_map,
                    feat_idx=self.model._conv_idx,
                    first_chunk=(not self.started),
                )
                self.started = True
                piece = unpatchify(out_i, patch_size=2).float().clamp_(-1, 1)
                out_list.append(piece)
                if on_partial is not None:
                    on_partial(piece.permute(0, 2, 1, 3, 4))  # [B, t, C, H, W]

            if not out_list:
                raise RuntimeError("Received empty latent chunk during streaming decode")

            out = torch.cat(out_list, dim=2).squeeze(0)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)  # [B, C, T, H, W]
        output = output.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        return output


class _AsyncStreamingVAEChunkDecoder:
    """`_StreamingVAEChunkDecoder` driven from a worker thread on a private CUDA stream.

    Why: on an H200 a 3-latent block costs ~460 ms of generator forwards and ~805 ms of VAE
    decode. Decoding inline serialises the two, so the block takes their sum (~1.3 s, 9 fps)
    when it could take the larger alone (~0.8 s, 15 fps). The decode is the same work either
    way -- it just no longer blocks the next window's denoising.

    Ordering is not optional: the Wan2.2 decoder keeps a causal cache across chunks, so chunk
    N+1's output depends on chunk N having already run. A single worker thread consuming a FIFO
    is what makes that safe -- do not widen it to a pool.

    Set WAN_ASYNC_VAE=0 to fall back to decoding inline (same numbers, for A/B).
    """

    def __init__(self, vae_wrapper: WanVAEWrapper, device=None,
                 on_chunk: Optional[Callable[[torch.Tensor, int], None]] = None,
                 vae_device=None):
        self._dec = _StreamingVAEChunkDecoder(vae_wrapper)
        self._on_chunk = on_chunk
        self._device = device
        # When the VAE lives on its own GPU, decode stops competing with the generator for SMs.
        # On one GPU the overlap only recovers the launch bubbles: measured 191 -> 346 ms per
        # generator call and 806 -> 1088 ms per decode when the two run concurrently. A block's
        # latents are ~0.45 MB, so shipping them to a second device is free by comparison.
        self._vae_device = torch.device(vae_device) if vae_device is not None else None
        self._enabled = (
            os.environ.get("WAN_ASYNC_VAE", "1") != "0"
            and torch.cuda.is_available()
            and device is not None
            and torch.device(device).type == "cuda"
        )
        self._work_device = self._vae_device or (torch.device(device) if device is not None else None)
        self._stream = torch.cuda.Stream(device=self._work_device) if self._enabled else None
        self._q: "queue.Queue" = queue.Queue()
        self._err: Optional[BaseException] = None
        self._lock = threading.Lock()

        # Results, in submit order.
        self.chunks: List[torch.Tensor] = []
        self.frames = 0
        self.first_frame_ts: Optional[float] = None
        self.first_chunk_frames = 0

        self._thread = None
        if self._enabled:
            self._thread = threading.Thread(target=self._run, name="vae-decode", daemon=True)
            self._thread.start()

    # -- producer side ---------------------------------------------------------------
    def submit(self, latent_chunk: torch.Tensor, block_index: int) -> None:
        """Hand a clean latent block over for decoding. Returns without waiting for it."""
        self._reraise()
        # Clone so the caller is free to overwrite its own buffers while we decode.
        chunk = latent_chunk.detach().clone()
        if not self._enabled:
            self._decode_one(chunk, block_index)
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self._device))
        self._q.put((chunk, int(block_index), event))

    def join(self) -> None:
        """Wait for every submitted block to finish decoding."""
        if self._enabled:
            self._q.put(None)
            self._thread.join()
        self._reraise()

    def clear(self) -> None:
        self._dec.clear()

    def _reraise(self) -> None:
        with self._lock:
            err = self._err
        if err is not None:
            raise err

    # -- worker side -----------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            chunk, block_index, event = item
            try:
                with torch.cuda.device(self._work_device), torch.cuda.stream(self._stream):
                    self._stream.wait_event(event)
                    if self._vae_device is not None and chunk.device != self._vae_device:
                        # Cross-device: we hold the only reference to `chunk` until the stream
                        # sync below, so the source cannot be freed out from under the copy.
                        chunk = chunk.to(self._vae_device, non_blocking=True)
                    else:
                        # Same device: the chunk was allocated on the caller's stream, so keep
                        # the allocator from recycling it while this stream still reads it.
                        chunk.record_stream(self._stream)
                    self._decode_one(chunk, block_index, sync_stream=True)
            except BaseException as exc:  # surfaced to the caller on the next submit/join
                with self._lock:
                    if self._err is None:
                        self._err = exc
                return

    def _decode_one(self, chunk: torch.Tensor, block_index: int, sync_stream: bool = False) -> None:
        def _deliver(piece: torch.Tensor) -> None:
            # Scale on the decode stream, then make sure it has landed before the piece leaves
            # this thread -- the consumer reads it on a different stream.
            piece = (piece * 0.5 + 0.5).clamp(0, 1)
            if sync_stream:
                self._stream.synchronize()
            self.chunks.append(piece)
            n = int(piece.shape[1])
            if n > 0:
                if self.first_frame_ts is None:
                    self.first_frame_ts = time.perf_counter()
                    self.first_chunk_frames = n
                self.frames += n
            if self._on_chunk is not None:
                self._on_chunk(piece, int(block_index))

        with torch.no_grad():
            self._dec.decode_latent_chunk(chunk, on_partial=_deliver)


