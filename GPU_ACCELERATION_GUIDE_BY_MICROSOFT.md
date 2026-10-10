---
layout: Conceptual
title: GPU accelerated ML training in WSL | Microsoft Learn
canonicalUrl: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
breadcrumb_path: /windows/wsl/breadcrumb/toc.json
uhfHeaderId: MSDocsHeader-Windows-DevTools
recommendations: true
feedback_product_url: https://github.com/microsoft/WSL/issues
feedback_system: OpenSource
ms.service: dev-environment
ms.subservice: windows-subsystem-for-linux
author: GrantMeStrength
ms.author: jken
ms.reviewer: crloewen
adobe-target: true
description: Learn how to setup the Windows Subsystem for Linux with NVIDIA CUDA, TensorFlow-DirectML, and PyTorch-DirectML. Read about using GPU acceleration with WSL to support machine learning training scenarios.
ms.date: 2024-11-19T00:00:00.0000000Z
ms.topic: get-started
locale: en-us
document_id: 30014605-59f7-d46f-fb2a-2d377e1b35dd
document_version_independent_id: 79e03484-4087-6c87-3a9f-9c0dab188fd2
updated_at: 2026-06-02T16:58:00.0000000Z
original_content_git_url: https://github.com/MicrosoftDocs/WSL/blob/live/WSL/tutorials/gpu-compute.md
gitcommit: https://github.com/MicrosoftDocs/WSL/blob/7b28cc1ee9b8ff672ada5e1c6c326d3573d703e5/WSL/tutorials/gpu-compute.md
git_commit_id: 7b28cc1ee9b8ff672ada5e1c6c326d3573d703e5
site_name: Docs
depot_name: WS.wsl
page_type: conceptual
toc_rel: ../toc.json
pdf_url_template: https://learn.microsoft.com/pdfstore/en-us/WS.wsl/{branchName}{pdfName}
feedback_help_link_type: ''
feedback_help_link_url: ''
word_count: 597
asset_id: tutorials/gpu-compute
moniker_range_name: 
monikers: []
item_type: Content
source_path: WSL/tutorials/gpu-compute.md
cmProducts:
- https://authoring-docs-microsoft.poolparty.biz/devrel/d6f38669-3d05-40f2-afd8-e49f7dd20884
- https://authoring-docs-microsoft.poolparty.biz/devrel/bcbcbad5-4208-4783-8035-8481272c98b8
- https://authoring-docs-microsoft.poolparty.biz/devrel/a3955c7b-f5ee-420d-aff5-d7119738f38b
spProducts:
- https://authoring-docs-microsoft.poolparty.biz/devrel/55265cca-de9b-48dc-a77a-9047cc39a575
- https://authoring-docs-microsoft.poolparty.biz/devrel/43b2e5aa-8a6d-4de2-a252-692232e5edc8
- https://authoring-docs-microsoft.poolparty.biz/devrel/b31948f4-2f38-404b-ac93-c3c8c5b3ae33
platformId: 0440fb53-1495-3093-7887-0e596cd57df7
---

# GPU accelerated ML training in WSL | Microsoft Learn

Machine learning (ML) is becoming a key part of many development workflows. Whether you're a data scientist, ML engineer, or starting your learning journey with ML the Windows Subsystem for Linux (WSL) offers a great environment to run the most common and popular GPU accelerated ML tools.

There are lots of different ways to set up these tools. For example, [NVIDIA CUDA in WSL](https://developer.nvidia.com/cuda/wsl), [TensorFlow-DirectML](https://pypi.org/project/tensorflow-directml/) and [PyTorch-DirectML](https://pypi.org/project/pytorch-directml/) all offer different ways you can use your GPU for ML with WSL. To learn more about the reasons for choosing one versus another, see [GPU accelerated ML training](/en-us/windows/ai/directml/gpu-accelerated-training).

This guide will show how to set up:

- NVIDIA CUDA if you have an NVIDIA graphics card and run a sample ML framework container
- TensorFlow-DirectML and PyTorch-DirectML on your AMD, Intel, or NVIDIA graphics card

## Prerequisites

- Ensure you are running [Windows 11](https://microsoft.com/software-download/windows11) or [Windows 10, version 21H2](https://microsoft.com/software-download/windows10) or higher.
- [Install WSL and set up a username and password for your Linux distribution](../install).

## Set up NVIDIA CUDA with Docker

1. [Download and install the latest driver for your NVIDIA GPU](https://www.nvidia.com/Download/index.aspx)
2. Install [Docker Desktop](wsl-containers#install-docker-desktop) or install the Docker engine directly in WSL by running the following command:

    ```bash
    curl https://get.docker.com | sh
    sudo service docker start
    ```
3. If you installed the Docker engine directly then install the NVIDIA Container Toolkit following the steps below.

    Set up the stable repository for the NVIDIA Container Toolkit by running the following commands:

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```

    Install the NVIDIA runtime packages and dependencies by running the commands:

    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    ```
4. Run a machine learning framework container and sample.

    To run a machine learning framework container and start using your GPU with this NVIDIA NGC TensorFlow container, enter the command:

    ```bash
    docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.03-tf2-py3
    ```

    ![TensorFlow with CUDA running inside a Docker container](https://user-images.githubusercontent.com/2146704/165866792-0fc1b9f6-a7be-49e2-8eb4-919a9b13a07c.png)

    You can run a pre-trained model sample that is built into this container by running the commands:

    ```bash
    cd nvidia-examples/cnn/
    python resnet.py --batch_size=64
    ```

    ![TensorFlow sample model training within Docker container](https://user-images.githubusercontent.com/2146704/165867329-fae2f8ec-e86d-412e-9e2c-dcfec0ec2429.gif)

Additional ways to get setup and utilize NVIDIA CUDA can be found in the [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl).

## Set up TensorFlow-DirectML or PyTorch-DirectML

1. Download and install the latest driver from your GPU vendors website: [AMD](https://www.amd.com/support/download/drivers.html), [Intel](https://downloadcenter.intel.com), or [NVIDIA](https://www.nvidia.com/drivers).
2. Setup a Python environment.

    We recommend setting up a virtual Python environment. There are many tools you can use to setup a virtual Python environment — for these instructions, we'll use [Anaconda's Miniconda](https://www.anaconda.com/docs/getting-started/miniconda).

    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    conda create --name directml python=3.7 -y
    conda activate directml
    ```
3. Install the machine learning framework backed by DirectML of your choice.

    TensorFlow-DirectML:

    ```bash
    pip install tensorflow-directml
    ```

    PyTorch-DirectML:

    ```bash
    sudo apt install libblas3 libomp5 liblapack3
    pip install torch-directml
    ```
4. Run a quick addition sample in an interactive Python session for [TensorFlow-DirectML](/en-us/windows/ai/directml/gpu-tensorflow-wsl#install-the-tensorflow-with-directml-package) or [PyTorch-DirectML](/en-us/windows/ai/directml/gpu-pytorch-wsl#install-the-pytorch-with-directml-package) to make sure everything is working.

If you have questions or run into issues, visit the [DirectML repo on GitHub](https://github.com/microsoft/DirectML#feedback).

## Multiple GPUs

If you have multiple GPUs on your machine you can also access them inside of WSL. However, you will only be able to access one at a time. To choose a specific GPU please set the environment variable below to the name of your GPU as it appears in device manager:

```bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME="<NameFromDeviceManager>"
```

This will do a string match, so if you set it to "NVIDIA" it will match the first GPU that starts with "NVIDIA".