## octa_superpixels_runner.py: 
## Convert each OCTA image together with the U-Net outputs (mask / optional probability map / optional uncertainty map) and 
## the ground-truth (GT) into superpixel-based graph data for training GATv2, and save it as .npz files.

## Output_uncertain_and_npy.py:
## Run inference on images using a trained U-Net model, and generate uncertainty maps and .npy files (in batch or for a single image) for use in SAM.py 
## to perform post-processing of the segmentation results.
