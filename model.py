import tracking
import training

red = (79, 83, 217)
blue = (202, 139, 66)

yranges = [(350, 680)]
scales = [1]
colors = [red]
hog_cells_per_step = 4
heat_threshold = 4
fir_length = heat_threshold*2 - 1
merge_boxes = True
expand_maximum = True

_, feature_extractor, classifier = training.load_model('./test.P')

tracker = tracking.CarTracker(yranges            = yranges,
                              scales             = scales,
                              colors             = colors,
                              hog_cells_per_step = hog_cells_per_step,
                              heat_threshold     = heat_threshold,
                              extract_features   = feature_extractor.extract_features,
                              predict            = classifier.predict,
                              fir_length         = fir_length,
                              merge_boxes        = merge_boxes,
                              expand_maximum     = expand_maximum,
                             )

