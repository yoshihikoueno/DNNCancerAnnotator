class TFRecordsType:
  regular = 'regular'
  ordered_slices = 'ordered_slices'
  input_3d = 'input_3d'


class TfExampleFields:
  patient_id = 'patient_id'
  slice_id = 'slice_id'

  image_file = 'image_file'
  image_encoded = 'image_encoded'

  annotation_file = 'annotation_file'
  annotation_encoded = 'annotation_encoded'

  label = 'label'

  examination_name = 'examination_name'

  width = 'width'
  height = 'height'
  depth = 'depth'
  image_3d_encoded = 'image_3d_encoded'
  annotation_3d_encoded = 'annotation_3d_encoded'


class InputDataFields:
  patient_id = 'patient_id'
  slice_id = 'slice_id'
  image_file = 'image_file'
  image_decoded = 'image_decoded'
  image_preprocessed = 'image_preprocessed'
  annotation_file = 'annotation_file'
  annotation_decoded = 'annotation_decoded'
  annotation_mask = 'annotation_mask'
  individual_masks = 'individual_masks'
  bounding_boxes = 'bounding_boxes'
  label = 'label'
  examination_name = 'examination_name'


class BoundingBoxFields:
  y_min = 'y_min'
  y_max = 'y_max'
  x_min = 'x_min'
  x_max = 'x_max'


class PickledDatasetInfo:
  seed = 'seed'
  patient_ids = 'patient_ids'
  file_names = 'file_names'
  patient_ratio = 'patient_ratio'
  data_dict = 'data_dict'
  patient_exam_id_to_num_slices = 'patient_exam_id_to_num_slices'


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
