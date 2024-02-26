from skimage import color, morphology, measure
import patchify
import numpy as np
import random
import openslide as osl
import os
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm
import torch
from PIL import Image
import shutil
import gc

def convert2gray(img):
    '''Convert rgb or rgba to grayscale; input must be rgb, rgba or already grayscale'''
    process_img = img.copy()
    process_img = np.array(process_img)
    img_shape = process_img.shape
    
    if len(img_shape) == 2:
        print('Image is already grayscale')
        return process_img
    
    # if uncommon number of dimensions image
    elif len(img_shape) >3:
        raise ValueError('Object parsed is not a 3D (h,w,c) style image - please check format')
    
    # if rgba, convert to rgb
    if img_shape[-1] == 4:
        process_img = color.rgba2rgb(process_img)
    
    # if img makes it here, it means it's rgb;
    # convert to gray and return
    return color.rgb2gray(process_img)

def get_tissue_mask(img, tissue_thresholds , object_size_threshold ):    
    process_img = convert2gray(img)
    tissue_img = (process_img >= tissue_thresholds[0]) & (process_img < tissue_thresholds[1])

    # remove excess tissue from outside of main tissue - erosion followed by removal of small objects
    tissue_img = morphology.binary_erosion(tissue_img)
    label_tissue = measure.label(tissue_img)
    tissue_img = morphology.remove_small_objects(label_tissue, min_size = object_size_threshold)
    # convert from label_mask
    tissue_img = (tissue_img > 0).astype(int)
    
    return tissue_img

def retrieve_req_downsample(sld_obj, req_mag, mag_property):
    try:
        # not the correct function to retrieve magnification from metadata
        start_mag = float(sld_obj.properties[mag_property])
        downsample = start_mag  / req_mag
        return float(downsample)
    except:
        print('Magnification',req_mag,'could not be extracted')

def retrieve_req_dims(sld_obj, req_downsample):
    original_dim = sld_obj.dimensions
    req_dims = [dim/req_downsample for dim in original_dim]
    return tuple(req_dims)

def retrieve_tissue_mask(sld_obj, tissue_thresholds = [0.2, 0.8], object_size_threshold= 5_000, req_mag = 1, mag_property = 'aperio.AppMag'):
    #1) Downsample slide to appropriate size for tissue threholding - trained at 1X
    tissue_down = retrieve_req_downsample(sld_obj, req_mag = req_mag, mag_property = mag_property)
    tissue_down_dims = retrieve_req_dims(sld_obj, tissue_down)
    tissue_thumb = sld_obj.get_thumbnail(tissue_down_dims)

    #2) process downsampled image to get tissue mask
    tissue_mask = get_tissue_mask(tissue_thumb, tissue_thresholds, object_size_threshold)

    return tissue_mask

def patch_mask(mask, patch_size):
    # Trim mask (so integer number of patches can be made), patch and linearise patches so in (n, h, w) shape

    # remove left and bottom cols / rows to make multiple of patch size (to allow reconstruction to original dims after patch / unpatch)
    orig_shape = mask.shape
    left_remove = orig_shape[0] % patch_size[0]
    bottom_remove = orig_shape[1] % patch_size[1]

    # trim mask 
    trim_mask = mask[left_remove:, bottom_remove:]

    # patchify mask 
    patches = patchify.patchify(trim_mask, patch_size, step = patch_size[0])

    # reshape so all patches in sequence - i.e. shape = [n, patch x, patch y] - easier to deal with
    reshaped_patches_shape = [patches.shape[0] * patches.shape[1]] + list(patch_size)
    patches_reshape = np.reshape(patches, reshaped_patches_shape)

    return patches_reshape, left_remove, bottom_remove, trim_mask.shape, patches.shape, patches

def unpatch_mask(patched_mask, left_remove, bottom_remove, trim_mask_shape, patches_shape):
    # Return patches to original image shape:
    # Reverse of above code i.e. unpatch, 

    # rebuild back to [rows, cols, patch x, patch y]
    patches_returned_shape = np.reshape(patched_mask, patches_shape)

    unpatched = patchify.unpatchify(patches_returned_shape, trim_mask_shape)

    # add back trimmed regions - left
    left_trim_add = np.zeros((left_remove, trim_mask_shape[1]))
    width_retrimmed = np.concatenate([left_trim_add, unpatched])
    
    # bottom
    bottom_trim_add = np.zeros((width_retrimmed.shape[0], bottom_remove))
    mask_retrimmed = np.concatenate([bottom_trim_add, width_retrimmed], axis = 1)

    return mask_retrimmed

def save_pil_region(pil_region_coord, params):
    wsi, outdir, wsi_name, axes_downsamples, wsi_region_size, out_patch_size, preceding_text = params
    # scale coordinates to original wsi dims

    pil_region_upsampled = [int(axes_downsamples[0]*pil_region_coord[0]), int(axes_downsamples[1]*pil_region_coord[1])]

    outfile = os.path.join(outdir, f'{preceding_text}{wsi_name}_{pil_region_upsampled}_{wsi_region_size}.png')
    
    # if outfile already exists, skip
    if os.path.exists(outfile) == False:

        # extract that region
        wsi_region = wsi.read_region(location = pil_region_upsampled, level = 0, size = wsi_region_size)
        
        # wsi_region = wsi_region.resize(out_patch_size)
        wsi_region = wsi_region.resize(out_patch_size).convert('RGB')

        wsi_region.save(outfile)
        # print(outfile, 'saved!')
    else:
        warnings.warn(f'{outfile} already exists. Skipping...')

def extract_downsampled_patches(wsi_file, 
                                mask, 
                                out_patch_size: tuple, 
                                outdir: str, 
                                req_mag: int = None, 
                                number_to_extract = 'all', 
                                preceding_text = '', 
                                positive_threshold: float = 0.9, 
                                max_mask_value = 1,  
                                mag_property = 'aperio.AppMag',
                                min_patch_number = 100):
    '''Extract downsampled patches from a wsi file. Regions to be extracted are passed via a mask where values > 0 = regions to be extracted.
    Tissue threshold = decimal amount of patch that needs to be covered to be accepted
    '''
     ## Read WSI and extract required number of regions from level 0
    wsi = osl.OpenSlide(wsi_file)
    
    # out_path_size is size required for output patches, taking into account downsampling
    # # Determine exact downsample between slide and mask on each axis
    axes_downsamples = [wsi.dimensions[i] / mask.shape[::-1][i] for i in range(2)]
    mask_patch_size = tuple([out_patch_size[i] / axes_downsamples[i] for i in range(2)])

    # if further downsample required for final image, increase mask patch size
    if req_mag is not None:
        req_downsample = retrieve_req_downsample(wsi, req_mag, mag_property)
        mask_patch_size = tuple([int(dim * req_downsample) for dim in mask_patch_size])

    # get ID No. of WSI 
    wsi_name = os.path.split(wsi_file)[-1] # remove path
    wsi_name = os.path.splitext(wsi_name)[0] # remove extension

    # patch the mask
    _, left_remove, bottom_remove, trim_mask_shape, patches_shape, patches = patch_mask(mask, mask_patch_size)

    ## determine which patches have full coverage
    coverage_patches = []

    # mask with full coverage to compare patches to
    compare_patch = np.ones(mask_patch_size) * max_mask_value

    # determine min positive content of patch 
    min_req_positive = sum(compare_patch.ravel()) * positive_threshold

    # over rows and columns of patched image, check if complete coverage
    # add row and col index if so
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            # check if patch has minimum required amount of positive region
            if sum(patches[i, j, :, :].ravel()) >= min_req_positive:
                coverage_patches.append([i,j])

    print(len(coverage_patches), f'tiles with {positive_threshold} coverage')
    if len(coverage_patches) >= min_patch_number:
        ## get coordinates, in terms of QA image pixels, instead of patch coordinates in grid
        covered_mask_coords = [[i * mask_patch_size[0], j * mask_patch_size[1]] for (i,j) in coverage_patches]

        # add pixels from left and bottom that had been trimmed - correct for this to prevent large errors in distance after scaling and openslide retrieval
        coverage_corrected = [[i + left_remove, j + bottom_remove] for (i,j) in covered_mask_coords]
        
        # if user has input limited number to extract, take random sample of the patches
        if number_to_extract != 'all':
            # check that available number is greater than requested number, if not, extract all available
            if len(coverage_corrected) < int(number_to_extract):
                print(f'Warning! Number of patches with requested coverage for wsi {wsi_file} is {len(coverage_corrected)} while {number_to_extract} requested.\nRetrieving all available.')
            
            # if more than requested masks, retrieve random sample of ones available
            else:
                coverage_corrected = random.sample(coverage_corrected, int(number_to_extract) )

        # convert np coords to PIL (reverse x and y)
        pil_region_coords = [mask_coords[::-1] for mask_coords in coverage_corrected]
        wsi_region_size = out_patch_size
        if req_mag is not None:
            # if no specific magnification required for output, just scale down to match patch size for mask
            wsi_region_size = tuple([int(req_downsample * wsi_region_size[i]) for i in range(2)])

        params = [wsi, outdir, wsi_name, axes_downsamples, wsi_region_size, out_patch_size, preceding_text]

        # save region for each region in list - single processed implementation
        for pil_region_coord in tqdm(pil_region_coords):
            save_pil_region(pil_region_coord, params)
    else:
        print(wsi_name, 'has fewer patches than minimum requested - skipping')

# Part 2: Use selected wsis to extract tissue patches 

def retrieve_patches(filepath:str, 
                    outdir = 'patches',
                    patch_size: tuple = (224, 224),
                    req_mag: int = 20,
                    positive_threshold: float = 1,
                    num_patches: int = 200,
                    transform=None, min_patch_number = 100):
    
    # make output and cache dirs
    cache_dir = os.path.join(outdir, 'cache')

    if os.path.exists(outdir) == False:
        os.makedirs(outdir, exist_ok = True)  

    if os.path.exists(cache_dir) == False:
        os.makedirs(cache_dir) 
        
    # retrieve tissue patches at 20X magnification
    sld = osl.OpenSlide(filepath)
    tissue_mask = retrieve_tissue_mask(sld)

    del sld

    extract_downsampled_patches(filepath, 
                                mask = tissue_mask,
                            out_patch_size = patch_size, 
                            outdir = cache_dir,
                            req_mag = int(req_mag), 
                            positive_threshold = positive_threshold, 
                            number_to_extract = num_patches,
                            min_patch_number = min_patch_number)
    
    # # clean
    # del tissue_mask
    # image_files = os.listdir(cache_dir)

    # if len(image_files) > 0:
    #     images = [Image.open(os.path.join(cache_dir, file)) for file in image_files]
    #     images = [transform(image) for image in images]
    #     images = torch.stack(images)

    #     # clean
    #     shutil.rmtree(cache_dir)
    #     gc.collect()

    #     return images, image_files
    
    # else:
    #     # clean
    #     shutil.rmtree(cache_dir)
    #     gc.collect()
    #     return None, None