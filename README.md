# Scalable Neural Video Representations with Learnable Positional Features (NVP)

Official PyTorch implementation of
["**Scalable Neural Video Representations with Learnable Positional Features**"](
https://arxiv.org/xxxxx) (NeurIPS 2022) by
[Subin Kim*](https://subin-kim-cv.github.io/)<sup>1</sup>,
[Sihyun Yu*](https://sihyun.me/)<sup>1</sup>,
[Jaeho Lee](https://jaeho-lee.github.io/)<sup>2</sup>,
and [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1</sup>.

<sup>1</sup>KAIST, <sup>2</sup>POSTECH

**TL;DR**: We propose a novel neural representation for videos that is the best of both worlds; achieved high-quality encoding and the compute-/parameter efficiency simultaneously. 

### [Project Page](https://subin-kim-cv.github.io/NVP) | [Paper](xxxx) | [Slide](https://subin-kim-cv.github.io/assets/2022_NVP/slide/kim2022NVP.pdf) 

<p align="center">
    <img src=figures/teaser_dynamic_compressed.gif width="900"> 
    <img src=figures/teaser_compressed.gif width="900"> 
</p>


## 1. Requirements
### Environments
Required packages are listed in `environment.yaml`.
Also, you should install the following packages:
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

pip install git+https://github.com/subin-kim-cv/tiny-cuda-nn/#subdirectory=bindings/torch
```
* This repository of [tiny-cuda-nn](https://github.com/subin-kim-cv/tiny-cuda-nn) is slightly different from original implementation of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

### Datasets
First, download the UVG-HD datasets from the following links:

* [UVG-HD](http://ultravideo.fi/#testsequences)

Then, extract RGB sequences from the original YUV videos of UVG-HD using [ffmpeg](https://ffmpeg.org/download.html). Here, `INPUT` is the input file name, and `OUTPUT` is a directory to save decompressed RGB frames.

```
ffmpeg -f rawvideo -vcodec rawvideo -s 1920x1080 -r 120 -pix_fmt yuv420p -i INPUT.yuv OUTPUT/f%05d.png
```

## 2. Training
Run the following script with a single GPU.

```train
CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_video.py --logging_root ./logs_nvp --experiment_name <EXPERIMENT_NAME> --dataset <DATASET> --num_frames <NUM_FRAMES> --config ./config/config_nvp_s.json 
```
* Option `--logging_root` denotes the path to save the experiment log.
* Option `--experiment_name` denotes the subdirectory to save the log files (results, checkpoints, configuration, etc.) existed under `--logging_root`.
* Option `--dataset` denotes the path of RGB sequences (e.g., `~/data/Jockey`).
* Option `--num_frames` denotes the number of frames to reconstruct (300 for the ShakeNDry video and 600 for other videos in UVG-HD).
* To reconstruct videos with 300 frames, please change the values of `t_resolution` in configuration file to 300.

## 3. Evaluation
Evaluation without compression of parameters (i.e., qunatization only)
```
CUDA_VISIBLE_DEVICES=0 python experiment_scripts/eval.py --logging_root ./logs_nvp --experiment_name <EXPERIMENT_NAME> --dataset <DATASET> --num_frames <NUM_FRAMES> --config ./logs_nvp/<EXPERIMENT_NAME>/config_nvp_s.json   
```
* Option `--save` denotes whether to save the reconstructed frames.
* One can specify an option `--s_interp` for a video superresolution results. It denotes the superresolution scale (e.g., 8).
* One can specify an option `--t_interp` for a video frame interpolation results. It denotes the temporal interpolation scale (e.g., 8).


Evaluation with compression of parameters using well-known image and video codecs.

1. Save the quantized parameters.
    ```
    CUDA_VISIBLE_DEVICES=0 python experiment_scripts/compression.py --logging_root ./logs_nvp --experiment_name <EXPERIMENT_NAME> --config ./logs_nvp/<EXPERIMENT_NAME>/config_nvp_s.json  
    ```
2. Compress the saved sparse positional image-/video-like features using codecs. 
    * Execute `compression.ipynb`. 
    * Please change the logging_root and experiment_name in `compression.ipynb` appropriately.
    * One can change `qscale`, `crf`, `framerate` which changes the compression ratio of sparse positinal features.
        * `qscale` ranges from 1 to 31, where larger values mean the worse quality (2~5 recommended).
        * `crf` ranges from 0 to 51 where larger values mean the worse quality (20~25 recommended).
        * `framerate` (25 or 40 recommended).


3. Evaluation with the compressed parameters.
    ```
    CUDA_VISIBLE_DEVICES=0 python experiment_scripts/eval_compression.py --logging_root ./logs_nvp --experiment_name <EXPERIMENT_NAME> --dataset <DATASET> --num_frames <NUM_FRAMES>  --config ./logs_nvp/<EXPERIMENT_NAME>/config_nvp_s.json --qscale 2 3 3 --crf 21 --framerate 25
    ```
    * Option `--save` denotes whether to save the reconstructed frames.
    * Please specify the option `--qscale`, `--crf`, `--framerate` as same with the values in the `compression.ipynb`.

## 4. Results
Reconstructed video results of NVP on UVG-HD, and other 4K/long/temporally dynamic videos are available at the following [project page](https://subin-kim-cv.github.io/NVP/).

Our model achieves the following performance on UVG-HD with a single NVIDIA V100 32GB GPU:

| Encoding Time  |   BPP  |    PSNR (&#8593;)  |    FLIP (&#8595;)  |    LPIPS (&#8595;) |
| -------------- | ------ | ------------------ | ------------------ | ------------------ |
|   ~5  minutes  |  0.901 | 34.57 $\pm$ 2.62   | 0.075 $\pm$ 0.021  | 0.190 $\pm$ 0.100  |
|   ~10 minutes  |  0.901 | 35.79 $\pm$ 2.31   | 0.065 $\pm$ 0.016  | 0.160 $\pm$ 0.098  |
|     ~1 hour    |  0.901 | 37.61 $\pm$ 2.20   | 0.052 $\pm$ 0.011  | 0.145 $\pm$ 0.106  |
|     ~8 hours   |  0.210 | 36.46 $\pm$ 2.18   | 0.067 $\pm$ 0.017  | 0.135 $\pm$ 0.083  |
* The reported values are averaged over the Beauty, Bosphorus, Honeybee, Jockey, ReadySetGo, ShakeNDry, and Yachtride videos in UVG-HD and measured using [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [FLIP](https://github.com/NVlabs/flip) repositories.

## Citation
```
```

## References
We used the code from following repositories: [SIREN](https://github.com/vsitzmann/siren), [Modulation](https://github.com/lucidrains/siren-pytorch), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
