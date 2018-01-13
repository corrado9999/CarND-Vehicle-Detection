import numpy as np
import cv2
from skimage.feature import hog
import sklearn.base

memory = sklearn.externals.joblib.Memory('./sklearn-cache/', verbose=False)

def convert_color(img, color_space='YCrCb'):
    conversion = dict(
        HLS   = cv2.COLOR_BGR2HLS,
        HSV   = cv2.COLOR_BGR2HSV,
        LAB   = cv2.COLOR_BGR2LAB,
        LUV   = cv2.COLOR_BGR2LUV,
        RGB   = cv2.COLOR_BGR2RGB,
        XYZ   = cv2.COLOR_BGR2XYZ,
        YCRCB = cv2.COLOR_BGR2YCrCb,
        YUV   = cv2.COLOR_BGR2YUV,
    )[color_space.upper()]
    return cv2.cvtColor(img, conversion)

@memory.cache
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, channel="ALL"):
    channels = range(3) if channel == "ALL" else [channel]
    return [
        hog(img[..., ch], orientations=orient,
            pixels_per_cell=(pix_per_cell,) * 2,
            cells_per_block=(cell_per_block,) * 2,
            transform_sqrt=False,
            visualise=vis, feature_vector=False,
            block_norm="L2-Hys",
           )
        for ch in channels
    ]

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def generate_windows(imshape,
                     window=64,
                     scale=1,
                     hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_cells_per_step=2,
):
    """Define a single function that can extract features using hog sub-sampling and make predictions.
    """
    imshape = np.array(imshape)
    new_shape = np.round(imshape / scale).astype(int)
    new_shape = (np.round(new_shape / hog_pix_per_cell) * hog_pix_per_cell).astype(int)
    yield new_shape

    # Define blocks and steps
    nblocks = (new_shape // hog_pix_per_cell) - hog_cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    win_draw = np.int(window * scale)
    nblocks_per_window = (window // hog_pix_per_cell) - hog_cell_per_block + 1
    nsteps = (nblocks - nblocks_per_window) // hog_cells_per_step + 1
    nysteps, nxsteps = nsteps

    for xb in range(int(nxsteps)):
        xpos = xb * hog_cells_per_step
        xleft = xpos * hog_pix_per_cell
        xleft_draw = np.int(xleft * scale)
        for yb in range(int(nysteps)):
            ypos = yb * hog_cells_per_step
            ytop = ypos * hog_pix_per_cell
            ytop_draw = np.int(ytop * scale)

            # Extract HOG for this patch
            hog_slices = (slice(ypos,  ypos  + nblocks_per_window),
                          slice(xpos,  xpos  + nblocks_per_window))
            img_slices = (slice(ytop,  ytop  + window),
                          slice(xleft, xleft + window))
            location = ((xleft_draw,            ytop_draw),
                        (xleft_draw + win_draw, ytop_draw + win_draw))
            yield hog_slices, img_slices, location

def adjust_scale(img, new_shape, interp_method=cv2.INTER_LINEAR):
    imshape = np.array(img.shape[:2]).astype(int)
    if np.any(imshape != new_shape):
        img = cv2.resize(img, tuple(new_shape[::-1]), 0, 0, interp_method)
    return img

def extract_features(img,
                     window=64,
                     scale=1, interp_method=cv2.INTER_LINEAR, cspace='RGB',
                     hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2,
                     hog_cells_per_step=2, hog_channel="ALL",
                     spatial_size=(32,32), hist_bins=32,
					 **_ #ignored, only for compatibility with FeatureExtraction class
):
    """Define a single function that can extract features using hog sub-sampling
    """
    windows = generate_windows(img.shape[:2], window, scale,
                               hog_orient, hog_pix_per_cell, hog_cell_per_block, hog_cells_per_step)
    img = convert_color(img, cspace)
    img = adjust_scale(img, next(windows))

    # Compute individual channel HOG features for the entire image
    hogs = get_hog_features(img, hog_orient, hog_pix_per_cell,
                                 hog_cell_per_block, channel=hog_channel)
    features, locations = [], []
    for hog_slices, img_slices, location in windows:
        # Extract the image patch
        subimg = img[img_slices[0], img_slices[1]]

        # Get color features
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)

        # Extract HOG for this patch
        hog_features = np.hstack(h[hog_slices].ravel()
                                 for h in hogs)

        # Accumulate features and locations
        features.append(np.hstack((spatial_features, hist_features, hog_features)))
        locations.append(location)

    return features, locations

class FeatureExtraction(sklearn.base.TransformerMixin):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_params(self, **params):
        self.__dict__.update(params)

    def get_params(self, deep=False):
        return self.__dict__.copy()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for img in X:
            feat, _ = extract_features(img, **self.__dict__)
            if len(feat) > 1:
                raise ValueError("More than one window was found, "
								 "please check your parameters")
            features.append(feat[0])
        return np.array(features)

    def extract_features(self, img, scale, hog_cells_per_step=None):
        kwargs = self.__dict__.copy()
        kwargs['scale'] = scale
        if hog_cells_per_step is not None:
            kwargs['hog_cells_per_step'] = hog_cells_per_step
        return extract_features(img, **kwargs)
