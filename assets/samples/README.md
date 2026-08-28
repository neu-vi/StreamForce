# Sample data

Six cases the offline inference scripts run on out of the box — three point-force and three
wind-force stills, at 832×480. They are the same images the interactive demo offers as gallery
presets (`demo/assets/`), with the same force values, so a case looks the same whether you
reach it through the demo or through `inference*.py`.

```
images/          point_1..3.png, wind_1..3.png   (832x480)
point_force.csv         3 rows   a single push
wind_force.csv          3 rows   a single global wind
point_force_change.csv  3 rows   the push reverses halfway
wind_force_change.csv   3 rows   the wind reverses halfway
```

Pick one with `--force_type`:

```bash
python inference_causal_rolling_forcing.py \
    --config_path configs/dmd_everything.yaml \
    --checkpoint_path <PATH_TO_DISTILLED_STUDENT_CKPT> \
    --force_type point_force \
    --output_folder outputs/
```

## What the columns mean

Forces are `0..1` magnitudes and degrees, matching what the demo's drag produces. The
inference scripts pin `min_force=0.0, max_force=1.0`, so the numbers are used as written
rather than renormalised against the file.

Angles are counter-clockwise from +x. Point-force anchors are pixel coordinates with **y
measured up from the bottom**, which is the convention the dataset expects (`coordy / height`
then flipped in the signal builder). The demo's `anchor_y` is measured down from the top, so
`coordy = height - anchor_y`.

| file | columns |
| :-- | :-- |
| `point_force.csv` | `image, angle, force, coordx, coordy, width, height, caption` |
| `wind_force.csv` | `image, wind_angle, wind_speed, width, height, caption` |
| `point_force_change.csv` | `image, angle1, force1, coordx1, coordy1, angle2, force2, coordx2, coordy2, width, height, change_at, caption` |
| `wind_force_change.csv` | `image, width, height, change_at, caption, wind_speed_1, wind_angle_1, wind_speed_2, wind_angle_2` |

`change_at` is a fraction of the clip (`0.5` = halfway). Drop the column and the change point
is sampled instead.

## Adding your own

Drop a 832×480 PNG in `images/` and add a row naming it. The loader globs `images/*.png` and
keeps only rows whose `image` matches a file present, so the four CSVs can share one folder
and a half-finished row is simply skipped.

The two `*_change.csv` files reuse the same six images: the second force is the first one
reversed (`angle + 180`), which is the mid-clip reversal the README describes. Change the
`*2` columns to make it something else.

## Note

The paper's numbers come from larger benchmark sets that are not distributed here. These six
cases are for checking that a checkpoint runs and behaves sensibly, not for reproducing
quantitative results.
