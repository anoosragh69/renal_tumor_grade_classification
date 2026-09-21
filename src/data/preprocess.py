"""
Task 6: CT Preprocessing Pipeline for KiTS19 Dataset
"""
import os
import numpy as np
import SimpleITK as sitk


def load_nifti(path):
    return sitk.ReadImage(path)


def apply_hu_windowing(image, lower=-400, upper=400):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower)
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(image)
    return result


def resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(image)


def extract_roi(image, segmentation, padding=10):
    seg_arr = sitk.GetArrayFromImage(segmentation)
    img_arr = sitk.GetArrayFromImage(image)
    coords = np.argwhere(seg_arr > 0)
    if len(coords) == 0:
        return image

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)

    img_shape = np.array(img_arr.shape)
    min_c = np.maximum(min_c - padding, 0)
    max_c = np.minimum(max_c + padding, img_shape)

    # Crop to valid range
    for i in range(3):
        if max_c[i] <= min_c[i]:
            max_c[i] = min(min_c[i] + 1, img_shape[i])

    cropped = img_arr[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
    result = sitk.GetImageFromArray(cropped)
    return result


def crop_or_pad(image, target_size=(128, 128, 128)):
    arr = sitk.GetArrayFromImage(image)
    pad_width = []
    for i, ts in enumerate(target_size):
        cs = arr.shape[i]
        pb = max(0, (ts - cs) // 2)
        pa = max(0, ts - cs - pb)
        pad_width.append((pb, pa))
    arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    for i, ts in enumerate(target_size):
        if arr.shape[i] > ts:
            start = (arr.shape[i] - ts) // 2
            arr = np.take(arr, range(start, start + ts), axis=i)
    return sitk.GetImageFromArray(arr)


def normalize_zscore(image):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m = np.mean(arr)
    s = np.std(arr)
    if s > 0:
        arr = (arr - m) / s
    return sitk.GetImageFromArray(arr)


def preprocess_ct_volume(imaging_path, segmentation_path=None,
                         target_size=(128, 128, 128), hu_lower=-400, hu_upper=400,
                         target_spacing=(1.0, 1.0, 1.0), use_segmentation=True):
    image = load_nifti(imaging_path)
    image = apply_hu_windowing(image, hu_lower, hu_upper)
    image = resample_to_isotropic(image, target_spacing, is_label=False)
    if use_segmentation and segmentation_path and os.path.exists(segmentation_path):
        seg = load_nifti(segmentation_path)
        seg = resample_to_isotropic(seg, target_spacing, is_label=True)
        image = extract_roi(image, seg)
    image = crop_or_pad(image, target_size)
    image = normalize_zscore(image)
    return sitk.GetArrayFromImage(image)


if __name__ == "__main__":
    DATA_DIR = "/mnt/SharedDrive/Projects/renal_tumor_grade_classification/kits19/data"
    case_id = "case_00000"
    imaging_path = os.path.join(DATA_DIR, case_id, "imaging.nii.gz")
    segmentation_path = os.path.join(DATA_DIR, case_id, "segmentation.nii.gz")
    print(f"Preprocessing {case_id}...")
    volume = preprocess_ct_volume(imaging_path, segmentation_path)
    print(f"Output shape: {volume.shape}")
    print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    print(f"Mean: {volume.mean():.2f}, Std: {volume.std():.2f}")
    print("Preprocessing pipeline test passed!")
