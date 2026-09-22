import os
import numpy as np
import SimpleITK as sitk

def create_fake_nifti(shape, is_mask=False):
    if is_mask:
        arr = np.zeros(shape, dtype=np.uint8)
        # Create a fake tumor in the center of the slices
        z, y, x = shape
        arr[z//2-2:z//2+2, y//2-20:y//2+20, x//2-30:x//2+30] = 1
    else:
        # Create fake CT data in HU
        arr = np.random.normal(loc=0, scale=100, size=shape).astype(np.int16)
        
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img

def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    kits19_data_dir = os.path.join(repo_root, "kits19", "data", "case_00000")
    os.makedirs(kits19_data_dir, exist_ok=True)
    
    img_path = os.path.join(kits19_data_dir, "imaging.nii.gz")
    seg_path = os.path.join(kits19_data_dir, "segmentation.nii.gz")
    
    shape = (10, 512, 512)
    img = create_fake_nifti(shape, is_mask=False)
    seg = create_fake_nifti(shape, is_mask=True)
    
    sitk.WriteImage(img, img_path)
    sitk.WriteImage(seg, seg_path)
    print(f"Created fake data at {kits19_data_dir}")

if __name__ == "__main__":
    main()
