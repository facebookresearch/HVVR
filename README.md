# HVVR
HVVR (Hierarchical Visibility for Virtual Reality) is an optimized software raycaster. It implements a hybrid CPU/GPU raycaster, suited for real-time rendering of effects such as lens distortion.

## Examples
modelviewer.vcxproj (vs2015/hvvr.sln) implements a simple model viewer, with optional #defines to enable depth of field and foveated rendering.

## Requirements
HVVR requires:
* Windows
* a CUDA-compatible GPU and the CUDA SDK (tested on GeForce GTX 1080 and Quadro P6000)
* (optional) the FBX SDK for importing content

## Building HVVR
0. Install CUDA SDK (and optionally FBX SDK)
1. Open vs2015/hvvr.sln
2. Select the "Develop/x64" configuration
3. Set "modelviewer" as the startup project
4. Build solution

## Installing HVVR
1. Download pre-built scene data files from: https://github.com/facebookresearch/HVVR/releases/latest
2. Copy pre-built scene data files (for example, sponza.bin) to HVVR/libraries/hvvr/samples_shared/data/scenes

## How HVVR works
To convert and load your own scenes:
1. Enable FBX import in HVVR/libraries/hvvr/samples_shared/model_import.h by setting MODEL_IMPORT_ENABLE_FBX to 1
2. Change the value of scenePath in gOnInit() in HVVR/projects/hvvr_samples/modelviewer/modelviewer.cpp to point at your FBX file
3. (Optionally, you can also export the FBX file to .bin for faster loading in the future, see the "MODEL_CONVERT" macro)

## Full documentation
...

## Join the HVVR community
See the CONTRIBUTING file for how to help out.

## License
HVVR is BSD-licensed. We also provide an additional patent grant.