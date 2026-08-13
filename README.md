# ReID metric learning

Images are discovered recursively below `train_data_path` and
`validation_data_path`. The identity is the part of each basename before the
first `_` (for example, `1_camera_02.jpg` has identity `1`). At least one
identity must have two images, and each dataset must contain at least two
identities. Identities with one image can still be sampled as negatives.

Edit `cfg/cfg.yaml`, then run:

```bash
python train.py --cfg cfg/cfg.yaml
```

`backbone` selects the embedding feature extractor. The default `compact_cnn`,
`mobilenet_v2`, `mobilenet_v3_small`, and `efficientnet_b0` are implemented
locally for edge conversion. They use convolution/depthwise-convolution,
LeakyReLU, lightweight residual Add, and global average pooling. He
initialization and residual paths prevent activations and gradients from
collapsing in the deeper models. BatchNorm, Normalization, Rescaling, and
squeeze-excitation layers are not used. These edge variants are initialized
randomly and are intentionally not exact replicas of the TensorFlow
Applications models.

Training uses PK batches: `identities_per_batch` identities and
`images_per_identity` images from each identity, with `batch_size` equal to
their product. For every image, batch-hard mining selects the farthest
same-identity embedding and the nearest different-identity embedding. The model
emits an `embedding_dim` vector and optimizes
`d(anchor, positive)^2 - min(d(anchor, negative), maximum_negative_distance)`.
This pulls positive pairs toward distance `0` and pushes negative pairs apart
until their distance reaches `maximum_negative_distance`.

When `use_id_classification_loss` is true, a training-only identity classifier
adds cross-entropy loss with weight `id_classification_loss_weight`. Classifier
weights are saved beside checkpoints for training resumption, while the saved
ReID model itself still outputs embeddings only. `id_label_smoothing` defaults
to `0.1` to reduce overconfidence on training identities.

Validation reports distance-based verification metrics at
`verification_threshold`: TAR, FRR, TNR, FAR, and the fraction of triplets for
which both the positive and negative are classified correctly. When the value
is null, the validation EER threshold is used for these diagnostic metrics. It
also reports distance percentiles, ROC-AUC, EER, and TAR operating points at
FAR 1%.

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

The best checkpoint is selected by the highest validation Rank-1 score and is
named with a `_rank1_<score>` suffix. Consequently, `query_data_path` is required
during training.

Every validation result is appended to `validation_log.csv` in the checkpoint
directory. The CSV contains the iteration, learning rate, and all reported
validation metrics so the complete training history remains available after
training finishes.

Every checkpoint directory contains the exact `cfg.yaml` used for that run,
including `embedding_dim` and `maximum_negative_distance`. Models are saved in
the HDF5 format with the `.h5` extension.

## Autonomous agent experiments

The repository-independent experiment loop is defined by two files:

- `EXPERIMENT.md`: committed project rules and successful experiment history.
- `agent_experiment_run.sh`: runs bounded agent experiments directly in the
  current repository.

Failed and inconclusive attempts are stored only in the ignored local file
`.experiment-history.md`. The selected agent reads it before choosing each new hypothesis so
the same failed experiment is not repeated.

Commit or otherwise remove every tracked and untracked working-tree change,
make sure the selected agent CLI is authenticated, and run:

```bash
chmod +x agent_experiment_run.sh
./agent_experiment_run.sh 10
```

The loop stops after three consecutive runs without an accepted commit. It
never performs automatic destructive cleanup; if an interrupted agent leaves
changes, the loop stops for manual inspection. Copy `EXPERIMENT.md` and
`agent_experiment_run.sh` to another repository and edit only the
project-specific section of `EXPERIMENT.md`.
