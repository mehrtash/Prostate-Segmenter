import SimpleITK as sitk
import numpy as np

from utils import reshape_volume, correct_exposure


class Segmenter(object):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def __predict(self, vol_nda):
        predicted_label = self.model.predict_test(input_test_slices_arr=vol_nda)
        predicted_label[predicted_label > self.threshold] = 1
        predicted_label[predicted_label <= self.threshold] = 0
        return predicted_label

    def segment_prostate_volume(self, input_volume_path, output_mask_path, rows, cols):
        vol = sitk.ReadImage(input_volume_path)
        vol = sitk.Cast(sitk.RescaleIntensity(vol), sitk.sitkUInt8)
        vol_nda = sitk.GetArrayFromImage(vol)
        original_shape = vol_nda.shape
        vol_nda = reshape_volume(vol_nda, rows, cols)
        vol_nda = np.expand_dims(vol_nda, axis=1)
        for i in range(len(vol_nda)):
            vol_nda[i] = correct_exposure(vol_nda[i])
        label_nda = self.__predict(vol_nda)
        label_nda = label_nda[:, 0, :, :]
        # TODO: Check the indices for shape!
        label_nda = reshape_volume(label_nda, original_shape[1], original_shape[2])
        label = sitk.GetImageFromArray(label_nda)
        label.CopyInformation(vol)
        writer = sitk.ImageFileWriter()
        writer.Execute(label, output_mask_path, True)
