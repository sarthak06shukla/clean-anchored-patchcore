# Ready-to-run reference banks

Each category directory contains the adaptive bank from budget 4,096, seed 23,
plus the original experiment configuration and fitting/calibration metadata.
The reference seed was fixed for demonstration; these are not test-selected
best models. Only `--method adaptive` is available in these compact model folders.

Use `code/predict.py --run models/bottle` (or `models/hazelnut`, `models/tile`)
with an image from the matching product category. The ImageNet backbone weights
are downloaded separately by TorchVision on first use.

These feature banks are derived from MVTec AD normal training images. Dataset
attribution and CC BY-NC-SA 4.0 terms are in `../THIRD_PARTY_NOTICES.md` and
`../data_provenance/dataset_license.txt`. All six methods and budgets can be
regenerated with the study runner.
