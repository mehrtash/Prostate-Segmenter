import getopt
import os
import sys

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prostatesegmenter')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

from prostatesegmenter.data import ProstateData
from prostatesegmenter.model import CNNModel
from prostatesegmenter.nets import unet1
from prostatesegmenter.segmenter import Segmenter


def main(argv):
    InputVolume = ''
    OutputLabel = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["InputVolume=", "OutputLabel="])
    except getopt.GetoptError:
        print('usage: fit.py -InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fit.py -InputVolume <InputVolumePath> --OutputLabel <OutputLabelPath>')
            sys.exit()
        elif opt in ("-i", "--InputVolume"):
            InputVolume = arg
        elif opt in ("-o", "--OutputLabel"):
            OutputLabel = arg
    if InputVolume == '' or OutputLabel == '':
        print('usage: fit.py -InputVolume <InputVolumePath> -OutputLabel <OutputLabelPath>')
        sys.exit()
    if os.path.isfile(InputVolume) and os.path.isdir(os.path.dirname(OutputLabel)):
        ds = ProstateData()
        print("Making the model.")
        model = unet1.model(weights=True, summary=False)
        cnn = CNNModel(data_streamer=ds, model=model)
        rows = 128
        cols = 128
        #
        print("Starting the segmenter.")
        sg = Segmenter(cnn)
        sg.segment_prostate_volume(InputVolume, OutputLabel, rows, cols)
    else:
        print("Make sure the input file exists and the output file directory is valid.")
        print("InputVolume: ", InputVolume)
        print("OutputLabel: ", OutputLabel)

if __name__ == "__main__":
    main(sys.argv[1:])
