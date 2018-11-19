class TfExampleFields:
  patient_id = 'patient_id'
  slice_id = 'slice_id'

  image_file = 'image_file'
  image_encoded = 'image_encoded'

  annotation_file = 'annotation_file'
  annotation_encoded = 'annotation_encoded'

  label = 'label'


class InputDataFields:
  patient_id = 'patient_id'
  slice_id = 'slice_id'
  image_file = 'image_file'
  image_decoded = 'image_decoded'
  annotation_file = 'annotation_file'
  annotation_decoded = 'annotation_decoded'
  annotation_mask = 'annotation_mask'
  label = 'label'


class PickledDatasetInfo:
  pickled_file_name = 'tfrecords_pickle'
  split_to_size = 'split_to_size'
  seed = 'seed'
  dataset_size = 'dataset_size'
  patient_ids = 'patient_ids'


class SplitNames:
  train = 'train'
  val = 'val'
  test = 'test'
  available_names = [train, val, test]


class PredictionFields:
  image_shape = 'image_shape'
  refined_box_encodings = 'refined_box_encodings'
  class_predictions_with_background = 'class_predictions_with_background'
  num_proposals = 'num_proposals'
  proposal_boxes = 'proposal_boxes'
  box_classifier_features = 'box_classifier_features'
  proposal_boxes_normalized = 'proposal_boxes_normalized'
  mask_predictions = 'mask_predictions'
  rpn_features_to_crop = 'rpn_features_to_crop'
  rpn_box_encodings = 'rpn_box_encodings'
  rpn_objectness_predictions_with_background = \
    'rpn_objectness_predictions_with_background'
  rpn_box_predictor_features = 'rpn_box_predictor_features'
  anchors = 'anchors'
