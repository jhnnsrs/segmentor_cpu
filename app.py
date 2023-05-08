from stardist.models import StarDist3D
from csbdeep.utils import Path, normalize
import sys
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import asyncio
from arkitekt import Arkitekt
from fakts.grants.remote.device_code import DeviceCodeGrant
from fakts.grants.remote.base import StaticDiscovery
from fakts import Fakts
from mikro.api.schema import (
    ModelFragment,
    from_xarray,
    RepresentationFragment,
    ContextFragment,
    get_image_image_links,
    LinkableModels,
    create_model,
    RepresentationVariety,
    ModelKind,
)
from rekuest.actors.functional import (
    CompletlyThreadedActor,
)
import numpy as np
from pydantic import Field
from arkitekt.tqdm import tqdm as atdqm
from arkitekt import easy
from stardist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from stardist import Rays_GoldenSpiral
from stardist.models import StarDist2D
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from tqdm import tqdm
import shutil
import uuid
from arkitekt import register
from enum import Enum
from typing import Optional
from concurrent.futures import ProcessPoolExecutor


class PreTrainedModels(str, Enum):
    STARDIST_ORGANOID_3D = "stardist3"
    STARDIST_STYLED = "stardist_styled"


active_model: Optional[ModelFragment] = None
active_stardist_model = None


def set_active_stardist_model(model: ModelFragment):
    global active_model, active_stardist_model
    if active_model:
        if active_model.id == model.id:
            return active_stardist_model
        
    active_model = model
    with model.data as f:
        shutil.unpack_archive(f, f".modelcache/{active_model.id}")
    active_stardist_model = active_model.id
    return active_stardist_model
            


def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y



@register()
def upload_pretrained(pretrained: PreTrainedModels) -> ModelFragment:
    """Upload pretrained Stardist

    Uploads a pretrained startdist model

    Args:
        pretrained (PreTrainedModels): The pretrained model to upload (see PreTrainedModels)

    Returns:
        ModelFragment: The uploaded model
    """
    archive = shutil.make_archive("active_model", "zip", f"models/{pretrained}")
    model = create_model(
        "active_model.zip",
        kind=ModelKind.TENSORFLOW,
        name=f"Segmentor Pretrained Model: {pretrained}",
        contexts=[],
    )

    return model




@register()
def predict_flou2(rep: RepresentationFragment) -> RepresentationFragment:
    """Segment Flou2

    Segments Cells using the stardist flou2 pretrained model

    Args:
        rep (Representation): The Representation.

    Returns:
        Representation: A Representation

    """
    print(f"Called wtih Rep {rep.data.nbytes}")
    assert rep.data.nbytes < 1000 * 1000 * 30 * 1 * 2, "Image is to big to be loaded"

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    axis_norm = (0, 1, 2)
    x = rep.data.sel(c=0, t=0, z=0).transpose(*"xy").data.compute()
    x = normalize(x)


    labels, details = model.predict_instances(x)

    array = xr.DataArray(labels, dims=list("xy"))

    nana = from_xarray(
        array,
        name="Segmented " + rep.name,
        origins=[rep],
        tags=["segmented"],
        variety=RepresentationVariety.MASK,
    )
    return nana



def run_predict(model_id, instance):
    active_stardist_model = StarDist3D(None, name=model_id, basedir=".modelcache")
    return active_stardist_model.predict_instances(instance, n_tiles=(1,8, 8))


@register()
def predict_stardist(
    rep: RepresentationFragment,
    model: ModelFragment,
) -> RepresentationFragment:
    """Predict Stardist

    Segments Cells using the stardist algorithm

    Args:
        rep (RepresentationFragment): The Image to segment.
        model (ModelFragment): The model to use for segmentation

    Returns:
        Representation: A Representation

    """
    print(f"Called wtih Rep {rep.data.nbytes}")

    # model = StarDist3D(name=random_dir)

    axis_norm = (0, 1, 2)
    x = rep.data.sel(c=0, t=0).transpose(*"zxy").data.compute()
    x = normalize(x, 1, 99.8, axis=axis_norm)

    model_id = set_active_stardist_model(model)

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_predict, model_id, x)
        labels, details = future.result()
        


    print("uploading")
    array = xr.DataArray(labels, dims=list("zxy"))

    nana = from_xarray(
        array,
        name="Segmented " + rep.name,
        origins=[rep],
        tags=["segmented"],
        variety=RepresentationVariety.MASK,
    )

    return nana
