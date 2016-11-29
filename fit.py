import os
import sys

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prostatesegmenter')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

from prostatesegmenter.data import ProstateData
from prostatesegmenter.model import CNNModel
from prostatesegmenter.nets import unet1
from prostatesegmenter.segmenter import Segmenter

if __name__ == '__main__':
    ds = ProstateData()
    model = unet1.model(weights=True, summary=True)
    cnn = CNNModel(data_streamer=ds, model=model)
    model_id = "unet1"
    uid = '2016_07_13_17_56_21' + '_' + model_id

    rows = 128
    cols = 128
    #
    sg = Segmenter(cnn, uid)
    image_file_name = '/home/mehrtash/github/DeepInfer-Client-build/input.nrrd'
    label_file_name = '/home/mehrtash/github/DeepInfer-Client-build/output.nrrd'
    sg.segment_prostate_volume(image_file_name, label_file_name, rows, cols)
