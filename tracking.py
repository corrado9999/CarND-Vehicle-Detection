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
                                                         scale, hog_cells_per_step)
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

def draw_labeled_bboxes(img, labels):
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
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_boxes(img, bboxes, colors=[(255, 0, 0)], thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for idx, (p1, p2) in bboxes:
        color = tuple(colors[min(len(colors)-1, idx)])
        cv2.rectangle(draw_img, tuple(p1), tuple(p2), color, thick)
    return draw_img

def clean_detections(image, box_list, heat_value=1, heat_threshold=1):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list, heat_value)

    # If many boxes intersect, expand the highest "heat" to the whole glob
    #labels, num_labels = scipy.ndimage.measurements.label(heat)
    #for slc in scipy.ndimage.find_objects(labels):
    #    heat[slc] = heat[slc].max()

    # Apply threshold to help remove false positives
    heat[heat <= heat_threshold] = 0

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = scipy.ndimage.measurements.label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img
