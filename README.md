# MARS-ventricles: Deep learning-based lateral ventricles segmentation

The **MIAC Automated Region Segmentation (MARS) for lateral ventricles** is a state-of-the-art, deep learning-based segmentation tool.

This repository includes ready-to-use, pre-built container image based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), along with the code needed to build the image.

> [!CAUTION]
> This method is **NOT a medical device** and **for non-commercial, academic research use only!** 
> Do NOT use the method for diagnosis, prognosis, monitoring or any other purposes in clinical use.

## TLDR;

Detailed instructions are provided below. Here is the quick start version:

```shell
# 1. Pull the container image and save as Apptainer .sif file 
apptainer build mars-ventricles.sif docker://ghcr.io/miac-research/ventricles-nnunet:latest

# 2. Run inference on a T1w image in the current working directory using Apptainer and GPU
apptainer run -B $(pwd) --nv mars-ventricles.sif T1.nii.gz
```

## Using the pre-built container images

The ready-to-use, pre-built image is available for download from the [Github container registry](https://github.com/miac-research/MARS-ventricles/packages). The image has been tested with Apptainer and Docker. Please refer to the instructions below for usage guidance.


### Data requirements

The segmentation requires **only one input, a 3D T1-weighted image** (e.g., MPRAGE, FSPGR, FFE) in NIfTI-1 data format. We recommend [dcm2niix](https://github.com/rordenlab/dcm2niix) for DICOM to NIfTI conversion. Importantly, the image must have been acquired **without the use of a contrast agent**. Check that the lateral ventricles are sufficiently covered by the field of view.  
The recommended resolution is 1 mm isotropic. Images with a different resolution will be resliced to 1mm isotropic before prediction, segmentation masks are returned in the original resolution. In case your image data deviates from 1 mm isotropic resolution, check the output carefully.

### Hardware requirements

While the inference can be run on CPU (>8 cores recommended), an NVIDIA GPU will greatly accelerate the calculation. The pre-built image supports a wide range of NVIDIA GPUs from compute capability 7.0 (Volta, 2017) to the most recent 12.0 (Blackwell, 2024).

### Using Apptainer

```shell
# 1. Pull the container image and save as .sif file 
apptainer build mars-ventricles.sif docker://ghcr.io/miac-research/ventricles-nnunet:latest

# 2. Run inference on a T1w image in the current working directory using GPU (flag "--nv")
apptainer run -B $(pwd) --nv mars-ventricles.sif T1.nii.gz

# For advanced usage, see available command line options:
apptainer run mars-ventricles.sif -h
```

### Using Docker

```shell
# 1. Pull the container image into your local registry
docker pull ghcr.io/miac-research/ventricles-nnunet:latest
docker tag ghcr.io/miac-research/ventricles-nnunet:latest mars-ventricles:latest

# 2. Run inference on a T1w image in the current working directory using GPU (flag "--gpus all")
docker run --rm --gpus all -v $(pwd):/data mars-ventricles:latest /data/T1.nii.gz

# For advanced usage, see available command line options:
docker run --rm mars-ventricles:latest -h
```

## Licenses of redistributed software

Please note the license terms of software components that we redistribute within our container images:

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet?tab=Apache-2.0-1-ov-file)

## Funding

Development and maintenance of this software is funded by the Medical Image Analysis Center (MIAC AG).

![MIAC Logo](images/miaclogo@2x.png)
