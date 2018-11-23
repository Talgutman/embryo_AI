from scipy.io import loadmat
import numpy as np

#takes the presumably best mask after post-proc (7th) and calcs volume
def mask_to_volume(masks_path, pix_height=1.56, pix_width=1.56, pix_depth=3):
    loaded_masks = loadmat(masks_path)  # loaded_masks is a dict
    ten_masks = loaded_masks['masks']
    best_mask = ten_masks[6,:,:,:]
    num_of_pix = np.sum(best_mask)
    volume = num_of_pix * pix_height * pix_width * pix_depth
    return volume


