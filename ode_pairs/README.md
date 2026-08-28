# Stage 2 — ODE solution pairs

`generate_ode_pairs.py` runs a **trained bidirectional teacher** over one force dataset and
saves, for every sample, five points along the teacher's denoising trajectory. Stage 3
(`trainer/ode.py`) regresses the causal student onto those trajectories, which is what gives
the 4-step student a sensible starting point before distillation.

This is a middle step, not an entry point: it needs a stage-1 checkpoint to exist, and its
output is consumed only by stage 3.

---

## Running it

```bash
torchrun --nproc_per_node=8 ode_pairs/generate_ode_pairs.py \
    --scenario point_diverse \
    --config_path configs/finetune_bidirectional_teacher.yaml \
    --checkpoint_path <PATH_TO_TEACHER_CKPT>
```

Run it from the **repository root** — dataset and config paths resolve against the working
directory. Output goes to `force_ode/<scenario>/` unless you pass `--output_folder`, alongside
`force_ode/<scenario>_videos/` holding one preview mp4 per sample with the force arrow drawn
on it (`--no_preview` skips the decode and the mp4s, which is a good deal faster).

Samples are sharded across ranks and **already-written `.pt` files are skipped**, so an
interrupted run resumes by simply relaunching it.

`--config_path` should be the config of the *teacher you are running*, normally
`configs/finetune_bidirectional_teacher.yaml` — it supplies `model_kwargs`, the negative
prompt, and `remove_carnation`. `--checkpoint_path` is the teacher's weights; without it you
get the raw Wan model, which is only useful for smoke-testing the plumbing.

---

## Scenarios

| `--scenario` | dataset | mode | stage-3 config key |
| :-- | :-- | :-- | :-- |
| `point_synthetic` | `point-force/train/point_force_23000` | video | `point_synthetic_force_data_path` |
| `point_change_synthetic` | `point-force-change-force` | video | `point_synthetic_force_change_data_path` |
| `wind_synthetic` | `wind-force/train/wind_force_15359` | video | `wind_synthetic_force_data_path` |
| `wind_change_synthetic` | `wind-force-change/wind_force_change_15000` | video | `wind_synthetic_force_change_data_path` |
| `point_diverse` | `point-force-diverse` | image | `point_diverse_force_data_path` |
| `point_change_diverse` | `point-force-diverse` (change CSV) | image | `point_diverse_force_change_data_path` |
| `wind_diverse` | `wind-force-diverse-16K-filtered_5835` | image | `wind_diverse_force_data_path` |
| `wind_change_diverse` | same root, change CSV | image | `wind_diverse_force_change_data_path` |

**video vs image mode.** Synthetic sources have ground-truth clips, so the
teacher is conditioned on the real first frame of a real video. The diverse sources are still
photographs — there is no ground-truth motion, and the teacher invents a plausible response to
the force. Both produce the same kind of `.pt`.

**`remove_carnation`.** Taken from the config and forwarded to every scenario, but it only
bites on one: `point_force_23000` is 12k `background_*` clips plus 11k `carnation_*` ones, and
is the only source with carnation rows. With `remove_carnation: true` (which the teacher config
sets) `point_synthetic` yields 12,000 samples rather than 23,000.

`--video_root_dir` / `--csv_path` override a scenario's paths.

---

## Wiring the output into stage 3

Each `.pt` is `{<index_key>: <row index into the source dataset>, "latents_list": tensor}`.
The index key is what tells stage 3 which row of which CSV a trajectory came from, so it
differs per family and **must not be renamed**:

| family | index key | read by |
| :-- | :-- | :-- |
| `*_diverse` | `diverse_index` | `Diverse*ODERegressionDataset` |
| `*_synthetic` | `synthetic_index` | `Synthetic*ODERegressionDataset` |

Point the matching `*_force_data_path` key in your stage-3 config at whatever
`--output_folder` you used, then run stage 3 normally:

```bash
torchrun --nproc_per_node=8 train.py --config_path configs/ode_everything.yaml
```

`ode_everything.yaml` expects all eight scenarios. Generating a full set means running this
script once per scenario — they are independent, so they can run back to back or on separate
nodes.

---

## Note

This replaces four near-identical scripts (~1,300 lines) that differed only in which dataset
they mounted — and two of which selected between sources by commenting blocks in and out.
The scenario table at the top of `generate_ode_pairs.py` is the whole of that difference.
