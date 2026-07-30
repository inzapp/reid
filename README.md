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
`d(anchor, positive)^2 - min(d(anchor, negative), maximum_negative_distance)`.
This pulls positive pairs toward distance `0` and pushes negative pairs apart
until their distance reaches `maximum_negative_distance`.

The embedding head uses Batch Normalization followed by a constant
`1/sqrt(embedding_dim)` rescaling. This keeps embedding scale stable and the
distance cap meaningful using conversion-friendly standard operations, without
per-vector L2 normalization.

Validation reports distance-based verification metrics at
`verification_threshold`: TAR, FRR, TNR, FAR, and the fraction of triplets for
which both the positive and negative are classified correctly. When the value
is null, the validation EER threshold is used for these diagnostic metrics. It
also reports distance percentiles, ROC-AUC, EER, and TAR operating points at
FAR 10%, 5%, and 1%.

Low-FAR measurements require a sufficiently large validation set. For example,
measuring FAR 0.01% with non-zero false matches requires at least 10,000
negative pairs; use a correspondingly large `--triplets` value when selecting
an operating threshold.

Evaluation embeds every validation image once in batches, then samples pair
indices and computes distances in chunks. `validation_pair_count` can therefore
be set to 100,000 or more without retaining triplet images in memory. When
camera IDs such as `_c1` are present in filenames, positive pairs are sampled
only across different cameras.

When `query_data_path` is configured, evaluation also reports a Market-1501
query-gallery Rank-1 score. Each query is searched against the validation gallery
after excluding gallery images with the same identity and camera as the query.
Rank-1 and pair-verification metrics both use only IDs greater than zero, so ID
`0000` distractors and negative-ID junk are excluded from the gallery.

Every checkpoint directory contains the exact `cfg.yaml` used for that run,
including `embedding_dim` and `maximum_negative_distance`. Models are saved in
the HDF5 format with the `.h5` extension.
