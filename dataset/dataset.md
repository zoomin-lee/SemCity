## Datasets
Datasets should have the following structure.

The triplane folder is created by `scripts/save_triplane.py` after `scripts/train_ae_main.py`.

### SemanticKITTI
You can download SemanticKITTI datasets from [here](http://www.semantic-kitti.org/assets/data_odometry_voxels_all.zip).

If you want to do semantic scene completion refinement, place the `.label` file from ssc method(e.g. [monoscene](https://github.com/astra-vision/MonoScene), [occdepth](https://github.com/megvii-research/OccDepth), [scpnet](https://github.com/SCPNet/Codes-for-SCPNet), [ssasc](https://github.com/jokester-zzz/ssa-sc)) in the following structure. 

    /dataset/
        └── sequences/
            ├── 00/
            |   ├── voxels/
            │   |     ├ 000000.label
            │   |     ├ 000000.invalid
            │   ├── monoscene/
            │   |     ├ 000000.label
            │   ├── occdepth/
            │   |     ├ 000000.label
            │   ├── scpnet/
            │   |     ├ 000000.label
            │   ├── ssasc/
            │   |     ├ 000000.label
            │   └── triplane/
            │         ├ 000000.npy
            │         ├ 000000_monoscene.npy
            │         ├ 000000_occdepth.npy
            │         ├ 000000_scpnet.npy
            │         ├ 000000_ssasc.npy
            ├── 01/
            .
            .
            └── 10/
        
### CarlaSC
You can download CarlaSC Cartesian datasets from [here](https://umich-curly.github.io/CarlaSC.github.io/download/).

The structure differs slightly from the original CarlaSC dataset to align with the SemanticKITTI dataset.
The `voxels` folder was originally the `evaluation` folder, which contains the GT for semantic scene completion.
    
    /carla/
        └── sequences/
            ├── Town01_Heavy/
            |   ├── voxels/
            │   |     ├ 000000.label
            │   |     ├ 000000.bin
            │   └── triplane/
            │         ├ 000000.npy
            ├── Town01_Medium/
            .
            .
            └── Town10_Light/