# ReID triplet training

Images are discovered recursively below `train_data_path` and
`validation_data_path`. The identity is the part of each basename before the
first `_` (for example, `1_camera_02.jpg` has identity `1`). At least one
identity must have two images, and each dataset must contain at least two
identities. Identities with one image can still be sampled as negatives.

Edit `cfg/cfg.yaml`, then run:

```bash
python train.py --cfg cfg/cfg.yaml
```

`batch_size` is the number of triplets per iteration. Each triplet contains two
different images of one randomly sampled identity and one image of a different
identity. The model emits an L2-normalized `embedding_dim` vector and optimizes
`max(d(anchor, positive) - d(anchor, negative) + distance_margin, 0)`.

Every checkpoint directory contains the exact `cfg.yaml` used for that run,
including `embedding_dim` and `distance_margin`.
