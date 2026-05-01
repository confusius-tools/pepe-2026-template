# fUSI mouse brain template from Pepe, Mariani *et al.*, 2026 

Export a ConfUSIus-loadable fUSI template aligned to the Allen atlas from the published
Pepe, Mariani *et al.* (2026) template.

## Source dataset

The input template used here comes from the dataset associated with:

- Pepe, C., Mariani, J.-C., Urosevic, M., Gini, S., Stuefer, A., Ricci, F.,
  Galbusera, A., Iurilli, G., and Gozzi, A. (2026). *Structural and dynamic
  embedding of the mouse functional connectome revealed by functional
  ultrasound imaging (fUSI).* DOI: `10.64898/2026.02.05.704055`.

The dataset is available on Zenodo at DOI `10.5281/zenodo.18486493`.

The published dataset is released under the Creative Commons Attribution 4.0
International License (CC BY 4.0). The template exported by this project is derived
from that dataset and should therefore be reused with appropriate attribution.

## Inputs

This repository contains the two input files required to build the exported
template:

- `inputs/published_params/source-BI_space-fUSI_desc-GillianTemplate_res-110umx100umx100um_feature.nii.gz`
- `inputs/registration/transform_Affine_AllenPIRRAS100um-2-fUSIPIRRAS_Composite.h5`

Notes:

- The template comes from the published dataset:
  `2026-02-03_PepeMariani_fUSI-anaesthetised/derivatives/Params/angio/templates/`.
- The HDF5 transform was communicated by the paper authors.

## Output

Running the export writes:

- `outputs/pepe-mariani-2026-fusi-template.nii.gz`

## Usage

Export the template:

```bash
uv run python main.py
```

Use it later with ConfUSIus:

```python
import confusius as cf
from confusius.atlas.atlas import Atlas

template = cf.load(
    "outputs/pepe-mariani-2026-fusi-template.nii.gz"
)
atlas = Atlas.from_brainglobe("allen_mouse_100um")
resampled_atlas = atlas.resample_like(
    template,
    template.affines["physical_to_sform"],
)
```
