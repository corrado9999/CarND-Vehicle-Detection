import numpy as np
import cv2
import scipy.ndimage.measurements

def find_cars(img, yranges, scales,
              extract_features, predict,
              hog_cells_per_step=None,
):
    box_list = []
    for n, (scale, (ystart, ystop)) in enumerate(zip(scales, yranges)):
        offset = np.array(((0, ystart), (0, ystart)))
        test_features, test_locations = extract_features(img[ystart:ystop],
                                                         scale,
                                                         hog_cells_per_step)
        if not len(test_features):
            continue
        #print("Predicting car/not car on %d windows" % len(test_features))
        test_prediction = predict(test_features)
        for i in test_prediction.nonzero()[0]:
            box_list.append((n, test_locations[i] + offset))
    return box_list

def add_heat(heatmap, bbox_list, value=1):
    # Iterate through list of bboxes
    for _, box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += value

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def get_bboxes_from_labels(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Append bounding box
        bboxes.append((np.inf, bbox))
    # Return the image
    return bboxes

def draw_boxes(img, bboxes, colors=[(255, 0, 0)], thick=6, labels=None):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for i, (idx, (p1, p2)) in enumerate(bboxes):
        color = tuple(colors[min(len(colors)-1, idx)])
        cv2.rectangle(draw_img, tuple(p1), tuple(p2), color, thick)
        if labels:
            cv2.putText(draw_img, labels[i], tuple(p1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255,)*3, 2, cv2.LINE_AA)
    return draw_img

def merge_boxes(image, box_list, expand_maximum=False, heat=None, heat_threshold=1):
    if heat is None:
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list, value=1)

    if expand_maximum:
        # If many boxes intersect, expand the highest "heat" to the whole glob
        labels, num_labels = scipy.ndimage.measurements.label(heat)
        for slc in scipy.ndimage.find_objects(labels):
             heat[slc] = heat[slc].max()

    # Apply threshold to help remove false positives
    heat[heat <= heat_threshold] = 0

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 15)

    # Find final boxes from heatmap using label function
    labels = scipy.ndimage.measurements.label(heatmap)
    bbox_heat = scipy.ndimage.measurements.maximum(heatmap,
                                                   labels[0],
                                                   index=range(1, labels[1]+1))
    bboxes = get_bboxes_from_labels(labels)

    return bboxes, bbox_heat, heat

class CarTracker(object):
    def __init__(self, extract_features, predict,
                 yranges=((400, 700),), scales=(1,), hog_cells_per_step=2,
                 colors=((32, 32, 192),), merge_boxes=True,
                 expand_maximum=True, heat_threshold=1,
                 alpha=0.9,
):
        self.extract_features = extract_features
        self.predict = predict
        self.yranges = yranges
        self.scales = scales
        self.hog_cells_per_step = hog_cells_per_step
        self.heatmap = None
        self.heat_threshold = heat_threshold
        self.alpha = alpha
        self.colors = colors
        self.expand_maximum = expand_maximum
        self.merge_boxes = merge_boxes

    def __call__(self, image, return_unmerged=False):
        image = image[..., [2,1,0]].copy()
        if self.heatmap is None:
            self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

        labels = None
        bboxes = find_cars(image, yranges=self.yranges, scales=self.scales,
                           hog_cells_per_step=self.hog_cells_per_step,
                           extract_features=self.extract_features,
                           predict=self.predict,
                          )
        if return_unmerged:
            unmerged = draw_boxes(image, bboxes, colors=self.colors)[..., [2,1,0]]

        if self.merge_boxes:
            bboxes, bbox_heat, self.heatmap = merge_boxes(
                   image, bboxes, self.expand_maximum,
                   heat=self.heatmap * self.alpha,
                   heat_threshold=self.heat_threshold)
            labels = ["%.0f" % x for x in bbox_heat]

        result = draw_boxes(image, bboxes, colors=self.colors, labels=labels)[..., [2,1,0]]
        if return_unmerged:
            return result, unmerged
        else:
            return result
