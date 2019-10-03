#!/usr/bin/env python

import os
import numpy as np
import argparse
import nibabel as nib

def get_wildcard(searchstring, printarray = False): # super dirty
	tmp_name = 'tmp_wildcard_%d' % np.random.randint(100000)
	os.system('echo %s > %s' % (searchstring, tmp_name))
	outstring = np.genfromtxt(tmp_name, dtype=str)
	os.system('rm %s' % tmp_name)
	if outstring.ndim == 0:
		return str(outstring)
	else:
		print("Multiple wildcards found ['%s' length = %d]. Outputting an array." % (searchstring, len(outstring)))
		if printarray:
			print (outstring)
		return outstring

def antsLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None):
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; antsRegistration -d 3 -r [ %s , %s, 1] -t Rigid[0.1] -m MI[ %s , %s , 1, 32] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox -o [%s%s_, %s%s.nii.gz] --float' % (numthreads,
																														reference,
																														mov,
																														reference,
																														mov,
																														outdir,
																														out_basename,
																														outdir,
																														out_basename))
	return ants_cmd

def antsNonLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None):
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; antsRegistration -d 3 --transform SyN[0.1,3,0] -m MI[ %s , %s , 1, 32] --convergence [100x70x50x20,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox -o [%s%s_, %s%s.nii.gz]' % (numthreads,
																								reference,
																								mov,
																								outdir,
																								out_basename,
																								outdir,
																								out_basename))
	return ants_cmd

def antsApplyTransformCmd(reference, mov, warps, outname, outdir = None):
	warps = np.array(warps)
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('antsApplyTransforms -d 3 -r %s -i %s -e 0 -o %s%s --float' % (reference,
																										mov,
																										outdir,
																										outname))
	for i in range(len(warps)):
		ants_cmd = ants_cmd + (' -t %s' % warps[i])
	return ants_cmd

def ants_bet(numthreads, reference_image, reference_image_brain_bn):
	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	T_template0 = "%s/static/ants_oasis_template_ras/T_template0.nii.gz" % scriptwd
	T_template0_BrainCerebellumProbabilityMask = "%s/static/ants_oasis_template_ras/T_template0_BrainCerebellumProbabilityMask.nii.gz" % scriptwd
	T_template0_BrainCerebellumRegistrationMask = "%s/static/ants_oasis_template_ras/T_template0_BrainCerebellumRegistrationMask.nii.gz" % scriptwd
	os.system("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; antsBrainExtraction.sh -d 3 -a %s -e %s -m %s -f %s -o %s" % (numthreads, 
																																									reference_image,
																																									T_template0,
																																									T_template0_BrainCerebellumProbabilityMask,
																																									T_template0_BrainCerebellumRegistrationMask,
																																									reference_image_brain_bn))



