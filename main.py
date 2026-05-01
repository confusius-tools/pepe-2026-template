from pathlib import Path

import confusius as cf
import numpy as np
import SimpleITK as sitk
import xarray as xr

ROOT = Path(__file__).parent
INPUTS_ROOT = ROOT / "inputs"
PUBLISHED_PARAMS_ROOT = INPUTS_ROOT / "published_params"
REGISTRATION_ROOT = INPUTS_ROOT / "registration"
OUTPUTS_ROOT = ROOT / "outputs"

FUSI_PATH = (
    PUBLISHED_PARAMS_ROOT
    / "source-BI_space-fUSI_desc-GillianTemplate_res-110umx100umx100um_feature.nii.gz"
)
COMPOSITE_PATH = (
    REGISTRATION_ROOT / "transform_Affine_AllenPIRRAS100um-2-fUSIPIRRAS_Composite.h5"
)
OUTPUT_TEMPLATE_PATH = OUTPUTS_ROOT / "pepe-mariani-2026-fusi-template.nii.gz"

# Extracted from the original moving image used to estimate the HDF5 registration.
A_ALLEN_PIRRAS = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# ConfUSIus stores world coordinates in component order (S, A, R) for an RAS NIfTI,
# while ITK uses LPS (x, y, z). This matrix converts between those conventions.
Q_CONFUSIUS_WORLD_TO_LPS = np.array(
    [
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)


def composite_to_4x4(transform: "sitk.Transform") -> np.ndarray:
    """Flatten a linear SimpleITK transform into a 4x4 affine.

    Parameters
    ----------
    transform : sitk.Transform
        Linear transform read from the ANTs HDF5 file.

    Returns
    -------
    numpy.ndarray
        Homogeneous affine in ITK's LPS world frame.
    """
    if isinstance(transform, sitk.CompositeTransform):
        out = np.eye(4)
        for idx in range(transform.GetNumberOfTransforms()):
            out = out @ composite_to_4x4(transform.GetNthTransform(idx))
        return out

    linear = np.asarray(transform.GetMatrix()).reshape(3, 3)  # type: ignore
    translation = np.asarray(transform.GetTranslation())  # type: ignore
    try:
        center = np.asarray(transform.GetCenter())  # type: ignore
    except AttributeError:
        center = np.zeros(3)

    out = np.eye(4)
    out[:3, :3] = linear
    out[:3, 3] = translation + center - linear @ center
    return out


def build_template_to_atlas_affine(fusi: xr.DataArray) -> np.ndarray:
    """Build the native template to BrainGlobe-atlas affine.

    Parameters
    ----------
    fusi : xarray.DataArray
        Published template loaded with ConfUSIus.

    Returns
    -------
    numpy.ndarray
        Pull affine mapping template physical coordinates to atlas physical
        coordinates.
    """
    transform = sitk.ReadTransform(str(COMPOSITE_PATH))
    t_lps = composite_to_4x4(transform)
    t_world = np.linalg.inv(Q_CONFUSIUS_WORLD_TO_LPS) @ t_lps @ Q_CONFUSIUS_WORLD_TO_LPS

    # AllenPIRRAS phys (R, I, P) -> BrainGlobe phys by permuting component order only.
    r_reorient = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    a_fusi = np.asarray(fusi.attrs["affines"]["physical_to_sform"])
    return r_reorient @ np.linalg.inv(A_ALLEN_PIRRAS) @ t_world @ a_fusi


def make_native_allen_oriented_template(
    fusi: xr.DataArray,
    template_to_atlas: np.ndarray,
) -> xr.DataArray:
    """Convert the native template into a ConfUSIus-style coronal layout.

    Parameters
    ----------
    fusi : xarray.DataArray
        Published template loaded with ConfUSIus.
    template_to_atlas : numpy.ndarray
        Pull affine mapping the native template physical space to BrainGlobe
        atlas physical space.

    Returns
    -------
    xarray.DataArray
        Reoriented template whose saved coordinates and `physical_to_sform`
        reconstruct the full atlas transform after `cf.save` / `cf.load`.
    """
    # Source file is stored on sagittal-like axes. For a ConfUSIus-style coronal layout
    # we want z=AP, y=IS, x=LR, which corresponds to z<-old x, y<-old y, x<-old z.
    linear_old_to_atlas = template_to_atlas[:3, :3]
    new_to_old_linear = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    linear_new_to_atlas = linear_old_to_atlas @ new_to_old_linear

    # ConfUSIus stores the physical origin in the coordinate arrays and the residual
    # orientation/shear in `physical_to_sform`.
    origin_zyx = template_to_atlas[:3, 3]

    affine = np.eye(4)
    affine[:3, :3] = linear_new_to_atlas
    affine[:3, 3] = (np.eye(3) - linear_new_to_atlas) @ origin_zyx

    z_step = float(abs(fusi.coords["x"].values[1] - fusi.coords["x"].values[0]))
    y_step = float(abs(fusi.coords["y"].values[1] - fusi.coords["y"].values[0]))
    x_step = float(abs(fusi.coords["z"].values[1] - fusi.coords["z"].values[0]))

    return xr.DataArray(
        np.asarray(fusi).transpose(2, 1, 0),
        dims=("z", "y", "x"),
        coords={
            "z": xr.DataArray(
                origin_zyx[0] + z_step * np.arange(fusi.shape[2]),
                dims=["z"],
                attrs={
                    "units": fusi.coords["x"].attrs.get("units", "mm"),
                    "voxdim": z_step,
                },
            ),
            "y": xr.DataArray(
                origin_zyx[1] + y_step * np.arange(fusi.shape[1]),
                dims=["y"],
                attrs={
                    "units": fusi.coords["y"].attrs.get("units", "mm"),
                    "voxdim": y_step,
                },
            ),
            "x": xr.DataArray(
                origin_zyx[2] + x_step * np.arange(fusi.shape[0]),
                dims=["x"],
                attrs={
                    "units": fusi.coords["z"].attrs.get("units", "mm"),
                    "voxdim": x_step,
                },
            ),
        },
        attrs={"affines": {"physical_to_sform": affine}, "sform_code": 1},
        name="fusi_template_native_allen_oriented",
    )


def export_template() -> Path:
    """Export the ConfUSIus-loadable template.

    Returns
    -------
    pathlib.Path
        Path to the written output NIfTI file.
    """
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    fusi = cf.load(str(FUSI_PATH))
    if "time" in fusi.dims:
        fusi = fusi.squeeze("time", drop=True)

    template_to_atlas = build_template_to_atlas_affine(fusi)
    exported = make_native_allen_oriented_template(fusi, template_to_atlas)
    cf.save(exported, OUTPUT_TEMPLATE_PATH)
    return OUTPUT_TEMPLATE_PATH


def main() -> None:
    output_path = export_template()
    loaded = cf.load(str(output_path))
    print(output_path)
    print(f"dims={loaded.dims} shape={loaded.shape}")


if __name__ == "__main__":
    main()
