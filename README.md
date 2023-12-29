# autoupscale
Automatic Upscaling with ComfyUI

## Setup
Install ComfyUI and the following add ons:
* UltimateSDUpscale
* ComfyMath
* Recommended Resolution Calculator

Install the following models:
* sd\_xl\_base\_1.0.safetensors checkpoint
* 4x\_foolhardy\_Remacri

## Running
To run the upscaler, you can specify a single file or a directory, in an
environment variable

```
UPSCALE_IMAGE=/path/to/folder/ python upscale.py
```
or
```
UPSCALE_IMAGE=/path/to/single/image.png python upscale.py
```

For a directory, every file in the directory will be upscaled

## Output
The output will be in /path/to/ComfyUI/output/[image\_name]\_upscaled\_[number].ext

## Workflow
The full workflow is available in image\_upscale.json to load into ComfyUI
