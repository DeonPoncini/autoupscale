import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    SaveImage,
    LoadImage,
    CLIPTextEncode,
    NODE_CLASS_MAPPINGS,
    CheckpointLoaderSimple,
)

def process_image(image_name):
    in_no_prefix = os.path.basename(os.path.splitext(image_name)[0])
    print("Upscaling: ", in_no_prefix)
    image_name_prefix = in_no_prefix + "_upscaled"
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_1 = loadimage.load_image(image=image_name)

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_2 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_4 = cliptextencode.encode(
            text="", clip=get_value_at_index(checkpointloadersimple_2, 1)
        )

        cliptextencode_5 = cliptextencode.encode(
            text="", clip=get_value_at_index(checkpointloadersimple_2, 1)
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_12 = upscalemodelloader.load_model(
            model_name="4x_foolhardy_Remacri.pth"
        )

        cm_nearestsdxlresolution = NODE_CLASS_MAPPINGS["CM_NearestSDXLResolution"]()
        cm_intbinaryoperation = NODE_CLASS_MAPPINGS["CM_IntBinaryOperation"]()
        recommendedrescalc = NODE_CLASS_MAPPINGS["RecommendedResCalc"]()
        ultimatesdupscale = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        saveimage = SaveImage()

        cm_nearestsdxlresolution_8 = cm_nearestsdxlresolution.op(
            image=get_value_at_index(loadimage_1, 0)
        )

        cm_intbinaryoperation_9 = cm_intbinaryoperation.op(
            op="Mul", a=get_value_at_index(cm_nearestsdxlresolution_8, 0), b=4
        )

        cm_intbinaryoperation_11 = cm_intbinaryoperation.op(
            op="Mul", a=get_value_at_index(cm_nearestsdxlresolution_8, 1), b=4
        )

        recommendedrescalc_6 = recommendedrescalc.calc(
            desiredXSIZE=get_value_at_index(cm_intbinaryoperation_9, 0),
            desiredYSIZE=get_value_at_index(cm_intbinaryoperation_11, 0),
        )

        ultimatesdupscale_3 = ultimatesdupscale.upscale(
            upscale_by=get_value_at_index(recommendedrescalc_6, 2),
            seed=random.randint(1, 2**64),
            steps=30,
            cfg=8,
            sampler_name="euler",
            scheduler="normal",
            denoise=0.25,
            mode_type="Linear",
            tile_width=1024,
            tile_height=1024,
            mask_blur=8,
            tile_padding=32,
            seam_fix_mode="None",
            seam_fix_denoise=1,
            seam_fix_width=64,
            seam_fix_mask_blur=8,
            seam_fix_padding=16,
            force_uniform_tiles=True,
            tiled_decode=False,
            image=get_value_at_index(loadimage_1, 0),
            model=get_value_at_index(checkpointloadersimple_2, 0),
            positive=get_value_at_index(cliptextencode_4, 0),
            negative=get_value_at_index(cliptextencode_5, 0),
            vae=get_value_at_index(checkpointloadersimple_2, 2),
            upscale_model=get_value_at_index(upscalemodelloader_12, 0),
        )

        saveimage_13 = saveimage.save_images(
            filename_prefix=image_name_prefix,
            images=get_value_at_index(ultimatesdupscale_3, 0),
        )

def main():
    import_custom_nodes()

    fn = os.environ['UPSCALE_IMAGE']
    if os.path.isdir(fn):
        for filename in os.scandir(fn):
            if filename.is_file():
                full_path=os.path.join(fn, filename)
                process_image(full_path)
    elif os.path.isfile(fn):
        process_image(fn)
    else:
        print('UPSCALE_IMAGE is not a file or directory')
        sys.exit()

if __name__ == "__main__":
    main()
