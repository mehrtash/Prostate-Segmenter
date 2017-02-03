import numpy as np
import os
import sys

import numpy as np
from progressbar import ProgressBar

from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve

def dice_coef(y_true, y_pred):
    import keras.backend as K
    K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def reshape_volume(nda_vol, rows, cols):
    from cv2 import resize, INTER_CUBIC
    nda_vol_reshaped = np.ndarray((nda_vol.shape[0], rows, cols), dtype=np.uint8)
    for i in range(len(nda_vol)):
        nda_vol_reshaped[i] = resize(nda_vol[i], (cols, rows), interpolation= INTER_CUBIC)
    return nda_vol_reshaped


def correct_exposure(nda):
    from skimage import exposure
    normalize_pctwise = (20, 95)
    pclow, pchigh = normalize_pctwise
    pl, ph = np.percentile(nda, (pclow, pchigh))
    nda = exposure.rescale_intensity(nda, in_range=(pl, ph))
    return nda


def download_file(fname, origin):
    datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, 'models')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)
    if os.path.exists(fpath):
        return fpath

    print('Downloading data from',  origin)

    global progbar
    progbar = None

    def dl_progress(count, block_size, total_size):
        global progbar
        if progbar is None:
            progbar = ProgressBar(max_value=total_size)
        elif count*block_size < total_size:
            progbar.update(count*block_size)
        else:
            progbar.finish()

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath, dl_progress)
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)
        raise e

    return fpath