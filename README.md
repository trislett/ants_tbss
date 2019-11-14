# ants_tbss
[![Build Status](https://travis-ci.org/trislett/ants_tbss.svg?branch=master)](https://travis-ci.org/trislett/ants_tbss)

TBSS (FSL) implementation with ANTs and T1w registration to template. ants_tbss creates the TBSS skeleton using ANTS without FA to FA registrations.

There are essentially two steps: (a) Inter-modality, intrasubject registration of the B0 image to subject T1w image (b) Registration of subject T1w image the MNI152 1mm brain template (by default). 

The software requires tight brain extractions for the T1w images. Brain extraction using [antsBrainExtraction.sh](https://github.com/ANTsX/ANTs/blob/master/Scripts/antsBrainExtraction.sh) and based on recommended settings from [fMRIprep](https://fmriprep.readthedocs.io/en/latest/workflows.html#brain-extraction-brain-tissue-segmentation-and-spatial-normalization).

It is also possible to use ants_tbss (--othermodality) for registration of other modalities such as fMRI (e.g., betted example_func.nii.gz) to B0, and use the previously calculated transformations to native space and standard space.

[voxel_slices](bin/voxel_slices) is also installed for fast production quality controls images.

_Autothresholded FA on T1w image in MNI_1mm space_
![Autothresholded FA on T1w image in MNI_1mm space](ants_tbss/static/FA_native.gif)


_FA with MNI_1mm brain image segmentation_
![FA with MNI_1mm brain image segmentation](ants_tbss/static/FA_meanT1.gif)


_Skeletonized FA on mean FA in MNI_1mm space_
![Skeletonized FA on mean FA in MNI_1mm space](ants_tbss/static/skelFA_stdFA.gif)


Citation:
Tustison NJ, Avants BB, Cook PA, Kim J, Whyte J, Gee JC, Stone JR. Logical circularity in voxel-based analysis: normalization strategy may induce statistical bias. Hum Brain Mapp. 2014 Mar;35(3):745-59. doi: 10.1002/hbm.22211.

Also read this post from the ANTS forum: https://sourceforge.net/p/advants/discussion/840261/thread/e6fc9a8c/

If you use the ants brain extraction script:

The script antsBrainExtraction.sh was used to perform brain extraction including N4BiasFieldCorrection and the OASIS template.

Tustison NJ, Avants BB, Cook PA, Zheng Y, Egan A, Yushkevich PA, Gee JC. N4ITK: improved N3 bias correction. IEEE Trans Med Imaging. 2010 Jun;29(6):1310–20. doi:10.1109/TMI.2010.2046908. 

## Temporary cookbook for the lazy

## Requirements
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), [ANTs](http://stnava.github.io/ANTs/), [parallel](https://www.gnu.org/software/parallel/)
* Python libraries: numpy, argparse, nibabel, matplotlib, scipy, scikit-image

## Installation

##### Create python environment (2.7 or 3.x)

```virtualenv -p python3.7 python37env```

```source python37env/bin/activate```

##### Clone the git page, and install ants_tbss

```pip install git+https://github.com/trislett/ants_tbss.git```

##### Make files list and run ants_tbss

Make a text file with betted B0 images

```for i in $(cat subjects); do echo /path/to/images/${i}*B0*nii.gz; done > B0_brain_list```

Make a text file with betted T1w images

```for i in $(cat subjects); do echo /path/to/images/${i}*T1w_T1_BrainExtractionBrain.nii.gz; done > T1w_brain_list```

Make a text file with betted B0 images

```for i in $(cat subjects); do echo /path/to/images/${i}*FA*nii.gz; done > FA_list```

Run ants_tbss

```ants_tbss --antsregtotemplate B0_brain_list T1w_brain_list --runtbss FA_list FA -nlws --numthreads 8```
