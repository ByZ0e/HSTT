## Overview
This repo contains source code for **Event Graph Guided Compositional Spatial-Temporal Reasoning for Video Question Answering**. In this work, propose a new learning framework, Hierarchical Spatial-Temporal Transformer for VideoQA (**HSTT**), to ground the multi-granularity visual concepts from the parsed **Event Graph** and combine them for compositional.

## Requirements 
We provide a Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and NVIDIA TITAN RTX.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.


## Getting Started

1. Download the testing code and data.
    All of them can be downloaded from Google Drive:
    `https://drive.google.com/file/d/1_dcy_XxkpFnlADar423eHxIYixvxs2QZ/view?usp=sharing`


2. Launch the Docker container for running the experiments.
    ```bash
    PATH_TO_STORAGE=/path/to/your/project/storage
    ```

    ```bash
    # docker image should be automatically pulled
    source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/video_db \
        $PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
    ```
    The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
    Note that the source code is mounted into the container under `/clipbert` instead 
    of built into the image so that user modification will be reflected without
    re-building the image. (Data folders are mounted into the container separately
    for flexibility on folder structures.)

3. Run inference.
    ```bash
    sh scripts/inference_star_demo.sh
    ```
    
    The results will be written under `$PATH_TO_STORAGE/finetune/star/results_action/step_8120_1_mean`.
    The file `submission_star_8120.json` will be generated for submission.
    You can submit it to the STAR challenge online evaluation leaderboard `https://eval.ai/web/challenges/challenge-page/1325/overview` for evaluation.

