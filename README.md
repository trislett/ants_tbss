# ants_tbss
TBSS (FSL) implementation with ANTs and T1w registration to template

Citation:
Tustison NJ, Avants BB, Cook PA, Kim J, Whyte J, Gee JC, Stone JR. Logical circularity in voxel-based analysis: normalization strategy may induce statistical bias. Hum Brain Mapp. 2014 Mar;35(3):745-59. doi: 10.1002/hbm.22211.

Also read this post from the ANTS forum: https://sourceforge.net/p/advants/discussion/840261/thread/e6fc9a8c/

Temporary cookbook for the lazy:

## Requirements
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), [ANTs](http://stnava.github.io/ANTs/), [parallel](www.gnu.org/software/parallel/)
* numpy, argparse, nibabel

## Installation

##### Create python environment (2.7 or 3.x)

```virtualenv -p python3.7 python37env```

```source python37env/bin/activate```

##### Clone the git page, and install ants_tbss

```git clone https://github.com/trislett/ants_tbss.git```

```cd ants_tbss```

```pip install .```

##### Make files list and run ants_tbss

Make a text file with betted B0 images

```for i in $(cat subjects); do echo /path/to/images/${i}*B0*nii.gz; done > B0_brain_list```

Make a text file with betted T1w images

```for i in $(cat subjects); do echo /path/to/images/${i}*T1w_T1_BrainExtractionBrain.nii.gz; done > T1w_brain_list```

Make a text file with betted B0 images

```for i in $(cat subjects); do echo /path/to/images/${i}*FA*nii.gz; done > FA_list```

Run ants_tbss

```ants_tbss --antsregtotemplate B0_brain_list T1w_brain_list --runtbss FA_list --numthreads 12```
