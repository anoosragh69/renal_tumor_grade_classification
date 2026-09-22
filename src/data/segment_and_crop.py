import os
import glob
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

def apply_hu_windowing(image_arr, lower=-400, upper=400):
    """Applies HU windowing and normalizes to [0, 1]."""
    arr = np.clip(image_arr, lower, upper)
    arr = (arr - lower) / (upper - lower)
    return arr

def crop_and_resize_slice(img_slice, seg_slice, target_size=(128, 128)):
    """
    Finds bounding box around tumor (seg > 0), crops image, and resizes.
    img_slice and seg_slice are 2D numpy arrays.
    Returns resized 2D image array and bounding box coords.
    """
    # Find tumor coordinates
    coords = np.argwhere(seg_slice > 0)
    if len(coords) == 0:
        return None, None

    # Get bounding box
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    # Crop image (add +1 to max bounds for slicing)
    cropped_img = img_slice[min_y:max_y+1, min_x:max_x+1]

    # Convert to PIL Image for resizing (LANCZOS)
    # Convert [0, 1] float to [0, 255] uint8 for PIL, or keep as float32
    # PIL can handle float32 ("F" mode), but resizing float32 works well
    pil_img = Image.fromarray(cropped_img)
    
    # Resize
    resized_img = pil_img.resize(target_size, resample=Image.Resampling.LANCZOS)
    
    return np.array(resized_img), (min_y, max_y, min_x, max_x)

def process_case(case_dir, output_dir):
    """Processes a single patient case."""
    patient_id = os.path.basename(case_dir)
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"Skipping {patient_id}: missing imaging or segmentation.")
        return

    # Load NIfTI volumes
    img_sitk = sitk.ReadImage(img_path)
    seg_sitk = sitk.ReadImage(seg_path)

    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg_sitk)
    
    # Apply HU windowing
    img_arr = apply_hu_windowing(img_arr)

    # Output patient directory
    pat_out_dir = os.path.join(output_dir, patient_id)
    os.makedirs(pat_out_dir, exist_ok=True)

    # Iterate through slices
    num_slices = img_arr.shape[0]
    saved_count = 0
    
    for slice_idx in range(num_slices):
        img_slice = img_arr[slice_idx]
        seg_slice = seg_arr[slice_idx]
        
        # Check if tumor exists in this slice
        if not np.any(seg_slice > 0):
            continue
            
        resized_img, _ = crop_and_resize_slice(img_slice, seg_slice)
        
        if resized_img is not None:
            # Save as numpy array (could also save as PNG/TIFF)
            out_filename = os.path.join(pat_out_dir, f"slice_{slice_idx:04d}.npy")
            np.save(out_filename, resized_img)
            saved_count += 1
            
    print(f"Processed {patient_id}: saved {saved_count} patches.")

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "kits19", "data")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim")
    
    os.makedirs(output_dir, exist_ok=True)
    
    case_dirs = sorted(glob.glob(os.path.join(data_dir, "case_*")))
    if not case_dirs:
        print(f"No cases found in {data_dir}.")
        return

    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        process_case(case_dir, output_dir)

if __name__ == "__main__":
    main()
