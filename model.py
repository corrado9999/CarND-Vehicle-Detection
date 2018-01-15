import tracking
import training

red = (79, 83, 217)
blue = (202, 139, 66)
green = (92, 184, 92)

yranges = [(400, 500), (350, 580)]
scales = [0.5, 1]
colors = [blue, green, red] #one more for the merged boxes
hog_cells_per_step = 2
heat_thresholds = (32, 70) #75 60 (24, 60) (34, 55-60) (16, 50) (16, 46)
fir_length = 12
merge_boxes = True

static_params = dict(
    #yranges = [(400, 500), (350, 580), (500, 680)],
    #scales = [0.5, 1, 2],
    #colors = [blue, red, green],
    #heat_thresholds = (3, 7),
    heat_thresholds = (1, 3),
)
_, feature_extractor, classifier = training.load_model('./liblinear.P')

tracker = tracking.CarTracker(yranges            = yranges,
                              scales             = scales,
                              colors             = colors,
                              hog_cells_per_step = hog_cells_per_step,
                              heat_thresholds    = heat_thresholds,
                              extract_features   = feature_extractor.extract_features,
                              predict            = classifier.predict,
                              fir_length         = fir_length,
                              merge_boxes        = merge_boxes,
                             )

