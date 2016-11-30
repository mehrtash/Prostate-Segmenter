import os
import sys, getopt

SEGMENTER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'prostatesegmenter')
print(SEGMENTER_PATH)
sys.path.insert(1, SEGMENTER_PATH)

from prostatesegmenter.data import ProstateData
from prostatesegmenter.model import CNNModel
from prostatesegmenter.nets import unet1
from prostatesegmenter.segmenter import Segmenter


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'usage: fit.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
    if inputfile == '' or outputfile == '':
        print 'usage: fit.py -i <inputfile> -o <outputfile>'
        sys.exit()
    if os.path.isfile(inputfile) and os.path.isdir(os.path.dirname(outputfile)):
        ds = ProstateData()
        model = unet1.model(weights=True, summary=True)
        cnn = CNNModel(data_streamer=ds, model=model)
        rows = 128
        cols = 128
        #
        sg = Segmenter(cnn)
        sg.segment_prostate_volume(inputfile, outputfile, rows, cols)
    else:
        print "Make sure the input file exists and the output file directory is valid."
        print "inputfile: ", inputfile
        print "outputfile: ", outputfile

if __name__ == "__main__":
    main(sys.argv[1:])
