#!/usr/bin/env python

import os
import numpy as np
import argparse
import nibabel as nib

assert "ANTSPATH" in os.environ, "The environment variable ANTSPATH must be declared."
ANTSPATH = os.environ['ANTSPATH']
assert "FSLDIR" in os.environ, "The environment variable FSLDIR must be declared."
FSLDIR = os.environ['FSLDIR']

def get_wildcard(searchstring, printarray = False): # super dirty
	"""
	Essentially glob but using bash. It outputs search arrays if more than one file is found.
	
	Parameters
	----------
	searchstring : str
	printarray : bool
	
	Returns
	-------
	outstring : str or array
	"""
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
	"""
	Wrapper for ANTs linear registration with some recommended parameters.
	Rigid transfomration: gradient step = 0.1
	Mutual information metric: weight = 1; bins = 32.
	Convergence: [1000x500x250x100,1e-6,10]
	shrink-factors: 8x4x2x1
	smoothing-sigmas: 3x2x1x0vox
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	reference : str
		The reference image.
	mov : str
		The moving image.
	out_basename : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; %s/antsRegistration -d 3 -r [ %s , %s, 1] -t Rigid[0.1] -m MI[ %s , %s , 1, 32] --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox -o [%s%s_, %s%s.nii.gz] --float' % (numthreads,
																														ANTSPATH,
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

	"""
	Wrapper for ANTs non-linear registration with some recommended parameters. I recommmend first using antsLinearRegCmd.
	SyN transformation: [0.1,3,0] 
	Mutual information metric: weight = 1; bins = 32.
	Convergence: [100x70x50x20,1e-6,10]
	shrink-factors: 8x4x2x1
	smoothing-sigmas: 3x2x1x0vox
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	reference : str
		The reference image.
	mov : str
		The moving image.
	out_basename : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; %s/antsRegistration -d 3 --transform SyN[0.1,3,0] -m MI[ %s , %s , 1, 32] --convergence [100x70x50x20,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox -o [%s%s_, %s%s.nii.gz]' % (numthreads,
																								ANTSPATH,
																								reference,
																								mov,
																								outdir,
																								out_basename,
																								outdir,
																								out_basename))
	return ants_cmd

def antsApplyTransformCmd(reference, mov, warps, outname, outdir = None):

	"""
	Wrapper for applying ANTs transformations (warps).
	
	Parameters
	----------
	reference : str
		The reference image.
	mov : str
		The moving image.
	warps : arr
		An array of warps to appy. It must always be an array even for a single warp!
	outname : str
		Output basename.
	outdir : str
		Output directory (options).

	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	warps = np.array(warps)
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('%s/antsApplyTransforms -d 3 -r %s -i %s -e 0 -o %s%s --float' % (ANTSPATH,
																										reference,
																										mov,
																										outdir,
																										outname))
	for i in range(len(warps)):
		ants_cmd = ants_cmd + (' -t %s' % warps[i])
	return ants_cmd

def ants_bet(numthreads, input_image, output_image_brain):
	"""
	Wrapper for applying ANTs transformations (warps).
	
	Parameters
	----------
	numthreads : int
		The number of threads for parallel processing
	input_image : str
		Anatomical image.
	output_image_brain : str
		Brain extracted output image.
	
	Returns
	-------
	ants_cmd : str
		Output of the command (that can be piped to os.system).
	"""

	scriptwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	be_template = "%s/static/ants_oasis_template_ras/T_template0.nii.gz" % scriptwd
	be_probability_mask = "%s/static/ants_oasis_template_ras/T_template0_BrainCerebellumProbabilityMask.nii.gz" % scriptwd
	be_registration_mask = "%s/static/ants_oasis_template_ras/T_template0_BrainCerebellumRegistrationMask.nii.gz" % scriptwd
	ants_cmd = ("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; %s/antsBrainExtraction.sh -d 3 -a %s -e %s -m %s -f %s -o %s" % (numthreads,
																																									ANTSPATH, 
																																									input_image,
																																									be_template,
																																									be_probability_mask,
																																									be_registration_mask,
																																									output_image_brain))
	return ants_cmd



