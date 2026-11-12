# Third-party attribution

## Dataset and derived images

MVTec AD image content is copyright 2019 MVTec Software GmbH and is supplied
under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
The dataset license is reproduced in `data_provenance/dataset_license.txt`.
The report figures, saved examples and demo overlays use resized/corrupted images
and heatmap overlays. All image-derived artifacts retain CC BY-NC-SA 4.0 and
attribution to MVTec.

The reference feature banks in `models/` are derived from MVTec AD normal images
and are provided for the same noncommercial research use with dataset attribution
and CC BY-NC-SA 4.0 terms. They do not contain the pretrained backbone weights.

Dataset: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

License: https://creativecommons.org/licenses/by-nc-sa/4.0/

## Method and model

The PatchCore framework is credited to Roth et al., CVPR 2022. Feature alignment
and pooling follow the documented reference design. No Amazon source files are
vendored. The project's selectors and experiment code are independently written.
The frozen pretrained WideResNet model is obtained through TorchVision.

## Document style

`paper/cvpr.sty` and `paper/ieeenat_fullname.bst` are unchanged files from the
CVPR 2026 author kit supplied with this project. Their original notices are
preserved. Using the style does not imply a CVPR submission or acceptance.
