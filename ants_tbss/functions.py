#!/usr/bin/env python

import os
import time
import datetime
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import scipy.misc as misc
from scipy import ndimage
from skimage import filters


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

def runCmd_log(cmd, logname = 'cmd_log'):
	"""
	Run a system command and logs it
	
	Parameters
	----------
	cmd : str
		Text string of the system command.
	logname : str
		The log file output file.
	Returns
	-------
	outstring : str or array
	"""
	ct = time.time()
	with open("cmd_log", "a") as logfile:
		logfile.write("[%s]\nCMD: %s\n" % (datetime.datetime.now(),cmd))
	os.system(cmd)
	print("Timestamp\t[%s]\tElapsed\t[%1.2fs]\n[CMD] %s" % (datetime.datetime.now(), (time.time() - ct), cmd))

def antsLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None, use_float = False):
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
	ants_cmd = ('''export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
						%s/antsRegistration -d 3 \
							-r [ %s , %s, 1] \
							-t Rigid[0.1] \
							-m MI[ %s , %s , 1, 32] \
							--convergence [1000x500x250x100,1e-6,10] \
							--shrink-factors 8x4x2x1 \
							--smoothing-sigmas 3x2x1x0vox \
							-o [%s%s_, %s%s.nii.gz]''' % (numthreads,
											ANTSPATH,
											reference,
											mov,
											reference,
											mov,
											outdir,
											out_basename,
											outdir,
											out_basename))
	if use_float:
		ants_cmd += ' --float'
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd

def antsNonLinearRegCmd(numthreads, reference, mov, out_basename, outdir = None, use_float = False):
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
	ants_cmd = ('''export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
					%s/antsRegistration -d 3 \
						--transform SyN[0.1,3,0] \
						-m MI[ %s , %s , 1, 32] \
						--convergence [100x70x50x20,1e-6,10] \
						--shrink-factors 8x4x2x1 \
						--smoothing-sigmas 3x2x1x0vox \
						-o [%s%s_, %s%s.nii.gz]''' % (numthreads,
															ANTSPATH,
															reference,
															mov,
															outdir,
															out_basename,
															outdir,
															out_basename))
	if use_float:
		ants_cmd += ' --float'
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd

def antsApplyTransformCmd(reference, mov, warps, outname, outdir = None, inverse = False, multipleimages = False):
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

	if multipleimages:
		e_ = '3'
	else:
		e_ = '0'

	warps = np.array(warps)
	if not outdir:
		outdir = ''
	else:
		if outdir[-1] != "/":
			outdir = outdir + "/"
	ants_cmd = ('%s/antsApplyTransforms -d 3 -r %s -i %s -e %s -o %s%s --float' % (ANTSPATH,
																										reference,
																										mov,
																										e_,
																										outdir,
																										outname))
	for i in range(len(warps)):
		if inverse:
			ants_cmd = ants_cmd + (' -t [%s, 1]' % warps[i])
		else:
			ants_cmd = ants_cmd + (' -t %s' % warps[i])
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd


def antsBetCmd(numthreads, input_image, output_image_brain):
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
	be_template = "%s/ants_tbss/ants_oasis_template_ras/T_template0.nii.gz" % scriptwd
	be_probability_mask = "%s/ants_tbss/ants_oasis_template_ras/T_template0_BrainCerebellumProbabilityMask.nii.gz" % scriptwd
	be_registration_mask = "%s/ants_tbss/ants_oasis_template_ras/T_template0_BrainCerebellumRegistrationMask.nii.gz" % scriptwd
	ants_cmd = ("export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; \
					%s/antsBrainExtraction.sh -d 3 \
						-a %s -e %s -m %s -f %s \
						-o %s" % (numthreads,
									ANTSPATH, 
									input_image,
									be_template,
									be_probability_mask,
									be_registration_mask,
									output_image_brain))
	ants_cmd = ants_cmd.replace('\t', '')
	return ants_cmd


def round_mask_transform(mask_image):
	"""
	Binarize a mask using numpy round and overwrites it.
	
	Parameters
	----------
	mask_image : str
		/path/to/mask_image
	
	Returns
	-------
	None
	"""
	img = nib.load(mask_image)
	img_data = img.get_data()
	img_data = np.round(img_data)
	nib.save(nib.Nifti1Image(img_data,img.affine), mask_image)


def nifti_to_float_precision(img_name):
	"""
	Converts nifti image to float32 precision.
	
	Parameters
	----------
	img_name : str
		/path/to/image
	
	Returns
	-------
	None
	"""
	img = nib.load(img_name)
	if img.get_data_dtype() != '<f4':
		img_data = img.get_data()
		nib.save(nib.Nifti1Image(img_data.astype(np.float32), img.affine), img_name)


def autothreshold(data, threshold_type = 'yen', z = 2.3264):
	"""
	Autothresholds data.
	
	Parameters
	----------
	data : array
		data array for autothresholding
	threshold_type : str
		autothresholding algorithms {'otsu' | 'li' | 'yen' | 'otsu_p' | 'li_p' | 'yen_p' | 'zscore_p'}. '*_p' calculates thresholds on only positive data.
		Citations:
			Otsu N (1979) A threshold selection method from gray-level histograms. IEEE Trans. Sys., Man., Cyber. 9: 62-66.
			Li C.H. and Lee C.K. (1993) Minimum Cross Entropy Thresholding Pattern Recognition, 26(4): 617-625
			Yen J.C., Chang F.J., and Chang S. (1995) A New Criterion for Automatic Multilevel Thresholding IEEE Trans. on Image Processing, 4(3): 370-378.
	z : float
		z-score threshold for using zscore_p
	
	Returns
	-------
	lthres : float
		The lower threshold.
	uthres : float
		The highier threshold.

	"""
	if threshold_type.endswith('_p'):
		data = data[data>0]
	else:
		data = data[data!=0]
	if data.size == 0:
		print("Warning: the data array is empty. Auto-thesholding will not be performed")
		return 0, 0
	else:
		if (threshold_type == 'otsu') or (threshold_type == 'otsu_p'):
			lthres = filters.threshold_otsu(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif (threshold_type == 'li')  or (threshold_type == 'li_p'):
			lthres = filters.threshold_li(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif (threshold_type == 'yen') or (threshold_type == 'yen_p'):
			lthres = filters.threshold_yen(data)
			uthres = data[data>lthres].mean() + (z*data[data>lthres].std())
		elif threshold_type == 'zscore_p':
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
			if lthres < 0:
				lthres = 0.001
		else:
			lthres = data.mean() - (z*data.std())
			uthres = data.mean() + (z*data.std())
		if uthres > data.max(): # for the rare case when uthres is larger than the max value
			uthres = data.max()
		return lthres, uthres


# These functions are for voxel slices

def draw_outline(img_png, mask_png, outline_color = [1,0,0,1], remove_mask = False):
	"""
	Create a read outline of a mask.
	
	Parameters
	----------
	img_png : str
		png file of image. e.g., brain.png
	mask_png : str
		png file of mask. e.g. brain_mask.png
	outline_color : array
		float array of the outline colour  and alpha {r,g,b,a} ranging from zero to one. The default if red [1,0,0,1]
	remove_mask : bool
		flag to delete mask_png
	
	Returns
	-------
	None
	"""
	from scipy.ndimage.morphology import binary_erosion
	img = mpimg.imread(img_png)
	mask = mpimg.imread(mask_png)
	#check mask
	mask[mask[:,:,3] != 1] = [0,0,0,0]
	mask[mask[:,:,3] == 1] = [1,1,1,1]
	mask[mask[:,:,0] == 1] = [1,1,1,1]
	index = (mask[:,:,0] == 1)
	ones_arr = index *1
	m = ones_arr - binary_erosion(ones_arr)
	index = (m[:,:] == 1)
	img[index] = outline_color
	if remove_mask:
		os.remove(mask_png)
	mpimg.imsave(img_png, img)

def outlay_png(img_png, outlay_png, remove_overlay = False):
	"""
	Uses the alpha values to overlay a png on another png.
	
	Parameters
	----------
	img_png : str
		png file of image. e.g., brain.png
	outlay_png : str
		png file of mask. e.g. overlap.png
	remove_overlay : bool
		flag to delete outlay_png
	
	Returns
	-------
	None
	"""
	img = mpimg.imread(img_png)
	img_overlap = mpimg.imread(outlay_png)
	img[img_overlap[:,:,-1]!=0] = img_overlap[img_overlap[:,:,-1]!=0]

	if remove_overlay:
		os.remove(outlay_png)
	mpimg.imsave(img_png, img)

def write_colorbar(threshold, input_cmap, name_cmap, outtype = 'png', transparent = True):
	"""
	Returns the coordinates of non-empty range of an image array (i.e., array with three dimensions)
	
	Parameters
	----------
	data : arr
		three dimensional array
	affine : arr
		[optional] apply the input affine transformation first.
	
	Returns
	-------
	x_minmax : arr
		array with x-axis minimum and maximum
	y_minmax : arr
		array with y-axis minimum and maximum
	z_minmax : arr
		array with z-axis minimum and maximum
	"""
	a = np.array([[threshold[0],threshold[1]]])
	plt.figure()
	plt.imshow(a, cmap=input_cmap)
	plt.gca().set_visible(False)
	cax = plt.axes([0.1, 0.1, 0.03, 0.8])
	plt.colorbar(orientation="vertical", cax=cax)
	plt.savefig("%s_colorbar.%s" % (os.path.basename(name_cmap), outtype),bbox_inches='tight', transparent = transparent)
	plt.close()

def nonempty_coordinate_range(data, affine = None):
	"""
	Returns the coordinates of non-empty range of an image array (i.e., array with three dimensions)
	
	Parameters
	----------
	data : arr
		three dimensional array
	affine : arr
		[optional] apply the input affine transformation first.
	
	Returns
	-------
	x_minmax : arr
		array with x-axis minimum and maximum
	y_minmax : arr
		array with y-axis minimum and maximum
	z_minmax : arr
		array with z-axis minimum and maximum
	"""
	nonempty = np.argwhere(data!=0)
	if affine is not None:
		nonempty_native = nib.affines.apply_affine(affine, nonempty)
		x_minmax = np.array((nonempty_native[:,0].min(), nonempty_native[:,0].max()))
		y_minmax = np.array((nonempty_native[:,1].min(), nonempty_native[:,1].max()))
		z_minmax = np.array((nonempty_native[:,2].min(), nonempty_native[:,2].max()))
	else:
		x_minmax = np.array((nonempty[:,0].min(), nonempty[:,0].max()))
		y_minmax = np.array((nonempty[:,1].min(), nonempty[:,1].max()))
		z_minmax = np.array((nonempty[:,2].min(), nonempty[:,2].max()))
	return (x_minmax,y_minmax,z_minmax)


def sym_pad_x(arr, max_size):
	"""
	Pads a 2D array with zeros
	
	Parameters
	----------
	arr : arr
		input numpy array
	max_size : arr
		largest size
	
	Returns
	-------
	arr : arr
		padded numpy array
	"""
	pad_size = int(np.divide((max_size - arr.shape[0]),2))
	pad_arr = np.zeros((pad_size, arr.shape[1]))
	arr = np.row_stack((pad_arr, arr))
	arr = np.row_stack((arr, pad_arr))
	return arr


def correct_image(img_name, b_transparent = True, rotate = None, flip = False, base_color = [0,0,0]):
	"""
	Remove black from png and over-writes it.
	
	Parameters
	----------
	img_name : str
		/path/to/*.img
	b_transparent : bool
		make the color set by base_color (default is black) transparent (alpha = 0).
	rotate : float
		[optional] rotate image by degrees.
	flip : bool
		[optional] flip the image on the y-axis.
	base_color : arr
		default = [0,0,0] or black
	
	Returns
	-------
	None
	"""
	img = misc.imread(img_name)
	if b_transparent:
		if img_name.endswith('.png'):
			rows = img.shape[0]
			columns = img.shape[1]
			if img.shape[2] == 3:
				img_flat = img.reshape([rows * columns, 3])
			else:
				img_flat = img.reshape([rows * columns, 4])
				img_flat = img_flat[:,:3]
			alpha = np.zeros([rows*columns, 1], dtype=np.uint8)
			alpha[img_flat[:,0]!=base_color[0]] = 255
			alpha[img_flat[:,1]!=base_color[1]] = 255
			alpha[img_flat[:,2]!=base_color[2]] = 255
			img_flat = np.column_stack([img_flat, alpha])
			img = img_flat.reshape([rows, columns, 4])
	if rotate is not None:
		img = ndimage.rotate(img, float(rotate))
	if flip:
		img = img[:,::-1,:]
	misc.imsave(img_name, img)


def write_padded_png(img_data, x_space, y_space, z_space, outname, vmin = None, vmax = None, cmap = None):
	"""
	write padded png
	
	Parameters
	----------
	img_data : array
		image array (3D)
	x_space : array
		linspace of x coordinates
	y_space : array
		linspace of y coordinates
	z_space : array
		linspace of z coordinates
	outname : str
		output name (should end with png)
	cmap : str
		[optional] color map to use.
	
	Returns
	-------
	None
	"""

	max_size = int(np.max(img_data.shape))

	x_array = []
	for x in x_space:
		x_array.append(np.swapaxes(img_data[int(x),:,::-1],0,1))
	x_row = np.concatenate(np.array(x_array), 1)
	x_row = sym_pad_x(x_row, max_size)

	if x_row.shape[0] < max_size:
		x_row = np.row_stack((x_row, np.zeros((x_row.shape[1]))))

	y_array = []
	for y in y_space:
		y_array.append(np.swapaxes(img_data[:,int(y),::-1],0,1))
	y_row = np.concatenate(np.array(y_array), 1)
	y_row = sym_pad_x(y_row,max_size)

	if y_row.shape[0] < max_size:
		y_row = np.row_stack((y_row, np.zeros((y_row.shape[1]))))

	z_array = []
	for z in z_space:
		z_array.append(np.swapaxes(img_data[:,::-1,int(z)],0,1))
	z_row = np.concatenate(np.array(z_array), 1)
	z_row = sym_pad_x(z_row,max_size)

	if z_row.shape[0] < max_size:
		z_row = np.row_stack((z_row, np.zeros((z_row.shape[1]))))

	mask_array = np.column_stack((x_row, y_row))
	mask_array = np.column_stack((mask_array, z_row))

	mpimg.imsave(outname, mask_array,
					vmin = vmin,
					vmax = vmax,
					cmap=cmap)


def linear_cm(c_start, c_end, c_mid = None, alpha = True, hide_lower = True, cmap_name = 'from_list'):
	"""
	Function to create linear lookup table
	
	Parameters
	----------
	c_start : array
		start colour [float([R]), float(G), float(B)]. Each value must range from 0 to 1.
	c_end : array
		end colour
	c_mid : array, optional
		middle colour
	alpha : bool
		create a column for alpha. Default = True.
	hide_lower : bool
		hide values below the threshold. Default = True
	cmap_name : str, optional
		Name of color map to create.
	
	Returns
	-------
	cmap : dict
		Matplotlib colormap
	"""
	cmap_array = np.zeros((256,3))
	if c_mid is not None:
		for i in range(3):
			cmap_array[0:128,i] = np.linspace(c_start[i], c_end[i],128)
			cmap_array[127:256,i] = np.linspace(c_end[i], c_mid[i],129)
	else:
		for i in range(3):
			cmap_array[:,i] = np.linspace(c_start[i],c_end[i],256)
	if alpha:
		cmap_array = np.column_stack((cmap_array, np.ones(len(cmap_array))))
		if hide_lower:
			cmap_array[0,-1] = 0
	return colors.ListedColormap(colors = cmap_array, name = cmap_name)

def cm_hide_lower(cmap_name):
	"""
	Add an alpha and sets the bottom color to zero
		
	Parameters
	----------
	cmap_name : string
		The name of the matplotlib cmap
	
	Returns
	-------
	cmap : dict
		Matplotlib colormap
	"""
	cmap = plt.cm.get_cmap(cmap_name)
	try:
		cmap_array = np.array(cmap.colors)
	except:
		cmap_array = cmap(np.linspace(0,1,256))
	if cmap_array.shape[1] == 1:
		cmap_array = np.column_stack((cmap_array, np.ones(len(cmap_array))))
	cmap_array[0,-1] = 0
	return colors.ListedColormap(colors = cmap_array, name = cmap_name)

