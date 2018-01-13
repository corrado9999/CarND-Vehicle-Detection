import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

import model
import features

plt.rcParams['figure.dpi'] = 50
plt.rcParams['figure.figsize'] = (30,)*2
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

def get_random_image(glob_expr):
    file_list = glob.glob(glob_expr)
    return cv2.imread(np.random.choice(file_list))

def plot_hogs(vehicle, non_vehicle, color_spaces=("RGB", "HLS", "YUV")):
    for cspace in color_spaces:
        plt.figure(figsize=(30, 20))
        for col, img, name in ((0, vehicle, "Car"), (2, non_vehicle, "Non-Car")):
            new = features.convert_color(img, cspace)
            for ch in range(len(cspace)):
                hog = features.get_hog_features(new, orient=9,
                                                     pix_per_cell=8,
                                                     cell_per_block=2,
                                                     vis=True,
                                                     channel=ch)[0][1]
                plt.subplot2grid((len(cspace), 4), (ch, col))
                plt.imshow(new[:, :, ch], cmap='gray')
                plt.title("%s %s[%s]" % (name, cspace, cspace[ch]))

                plt.subplot2grid((len(cspace), 4), (ch, col + 1))
                plt.imshow(hog, cmap='gray')
                plt.title("%s HoG-%s[%s]" % (name, cspace, cspace[ch]))
        plt.tight_layout()
        plt.show()

def plot_windows(*images, hog_cells_per_step=(4, 8), colors=[model.blue, model.red]):
    yrange = model.yranges[0]
    offset = np.array(((0, yrange[0]), (0, yrange[0])))
    for image in images:
        result = image
        for step, color in zip(hog_cells_per_step, colors):
            windows = features.generate_windows(image[yrange[0]:yrange[1]].shape[:2],
                                                hog_cells_per_step=step)
            next(windows) #discard first, it is the resampled shape
            result = tracking.draw_boxes(
                result, [(0, w[2] + offset) for w in windows],
                colors=[color], thick=2,
            )
        plt.imshow(result[..., ::-1])
        plt.show()

def plot_result(image=None, name=None, use_heatmap=False):
    if isinstance(image, str):
        path = image
        image = cv2.imread(path)[..., ::-1]
        if name is None:
            name = path
    else:
        if name is None:
            name = 'in-memory'

    model.tracker.reset()
    if use_heatmap:
        model.tracker.expand_maximum = False

    result, unmerged = model.tracker(image, return_unmerged=True)
    titles = ("%s (original boxes)" % name,
              "%s (merged boxes)" % name)
    if use_heatmap:
        result, unmerged = model.tracker.heatmap, result
        titles = (titles[1], "%s (heatmap)" % name)

    plt.subplot(121)
    plt.imshow(unmerged)
    plt.title(titles[0])

    plt.subplot(122)
    plt.imshow(result, cmap='hot', vmin=0, vmax=10)
    plt.title(titles[1])

    plt.tight_layout()
    plt.show()

def plot_frames(path, start, n_frames=6, use_heatmap=False):
    clip = VideoFileClip(path)
    for second in np.arange(n_frames)/clip.fps + start:
        frame = clip.get_frame(second)
        plot_result(frame, "%s@%.2fs" % (path, second), use_heatmap)
