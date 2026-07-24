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
identity. The model emits an `embedding_dim` vector and optimizes
`d(anchor, positive)^2 + max(distance_margin - d(anchor, negative), 0)^2`.
This directly pulls positive pairs toward distance `0` and penalizes negative
pairs only while their distance is below `distance_margin`.

Every checkpoint directory contains the exact `cfg.yaml` used for that run,
including `embedding_dim` and `distance_margin`. Models are saved in the HDF5
format with the `.h5` extension.
