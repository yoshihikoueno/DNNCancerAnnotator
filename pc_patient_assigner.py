import argparse
import os
import pickle

import numpy as np

from utils import standard_fields

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir',
                    help="Dataset Directory")
parser.add_argument('train_ratio', help='Should be integer between 1 and 100.')
parser.add_argument('val_ratio', help='Should be integer between 1 and 100.')
parser.add_argument('test_ratio', help='Should be integer between 1 and 100.')
parser.add_argument('--only_cancer', action='store_true')
parser.add_argument('--seed', required=True, type=int)

args = parser.parse_args()


def load_from_folder(folder, dataset_folder, id_prefix, class_label,
                     annotation_folder=None):
  assert ((class_label == 0 and annotation_folder is None) or (class_label == 1
          and annotation_folder is not None))

  images = dict()
  num_images = 0

  patient_folders = os.listdir(folder)
  num_patients = len(patient_folders)

  for patient_folder in patient_folders:
    patient_id = id_prefix + patient_folder
    assert(patient_id not in images)
    images[patient_id] = []
    patient_folder = os.path.join(folder, patient_folder)

    examination_folders = os.listdir(patient_folder)

    for examination_folder in examination_folders:
      examination_folder = os.path.join(patient_folder, examination_folder)
      assert(os.path.isdir(examination_folder))
      for f in os.listdir(examination_folder):
        f = os.path.join(examination_folder, f)
        if annotation_folder is not None:
          annotation_file = os.path.join(
            annotation_folder, os.path.basename(patient_folder),
            os.path.basename(examination_folder),
            os.path.basename(f).split('.')[0] + '.png')
          if not os.path.exists(annotation_file):
            raise ValueError("{} has no annotation file {}".format(
              f, annotation_file))
        else:
          annotation_file = f

        # Shorten the file paths, so that they are also valid in docker
        # environments
        annotation_file = annotation_file[len(dataset_folder) + 1:]
        image_file = f[len(dataset_folder) + 1:]

        images[patient_id].append(
          [image_file, annotation_file, class_label, patient_id,
           int(os.path.basename(f).split('.')[0]),
           os.path.basename(examination_folder)])
        num_images += 1

  return images, num_images, num_patients


# Copies part of the data dict consisting of the keys ids into a new dict
def _make_dataset_split(data, ids):
  return dict((k, data[k]) for k in ids)


def _make_dataset_splits(data, train_ratio, val_ratio, test_ratio):
  patient_ids = list(data.keys())
  np.random.shuffle(patient_ids)

  train_nb = int(np.floor(len(patient_ids) * train_ratio))
  val_nb = int(np.floor(len(patient_ids) * val_ratio))

  train_ids = patient_ids[:train_nb]
  val_ids = patient_ids[train_nb:train_nb + val_nb]
  test_ids = patient_ids[train_nb + val_nb:]

  train_split = _make_dataset_split(data, train_ids)
  val_split = _make_dataset_split(data, val_ids)
  test_split = _make_dataset_split(data, test_ids)

  return train_split, val_split, test_split


# Assign patients to a dataset split
def assign_patients(dataset_folder, train_ratio, val_ratio, test_ratio,
                    only_cancer, seed):
  output_file = os.path.join(
    dataset_folder, 'patient_assignment_{}_{}_{}_{}'.format(
      train_ratio, val_ratio, test_ratio, seed))

  train_ratio = float(train_ratio) / 100.0
  val_ratio = float(val_ratio) / 100.0
  test_ratio = float(test_ratio) / 100.0

  healthy_cases_folder = os.path.join(dataset_folder, 'healthy_cases')
  cancer_cases_folder = os.path.join(dataset_folder, 'cancer_cases')
  cancer_annotations_folder = os.path.join(dataset_folder,
                                           'cancer_annotations')

  assert(os.path.exists(healthy_cases_folder))
  assert(os.path.exists(cancer_cases_folder))
  assert(os.path.exists(cancer_annotations_folder))

  if only_cancer:
    healthy_images = dict()
    num_healthy_images = 0
    num_healthy_patients = 0
  else:
    healthy_images, num_healthy_images, num_healthy_patients = (
      load_from_folder(healthy_cases_folder, dataset_folder=dataset_folder,
                       id_prefix='h', class_label=0))

  cancer_images, num_cancer_images, num_cancer_patients = (
    load_from_folder(cancer_cases_folder, dataset_folder=dataset_folder,
                     id_prefix='c', class_label=1,
                     annotation_folder=cancer_annotations_folder))

  print("Healthy patients (images): {} ({})".format(
    num_healthy_patients, num_healthy_images))
  print("Cancer patients (images): {} ({})".format(
    num_cancer_patients, num_cancer_images))

  if num_healthy_patients == 0:
    patient_ratio = 1
  else:
    patient_ratio = float(num_healthy_patients) / float(num_cancer_patients)

  healthy_train, healthy_val, healthy_test = _make_dataset_splits(
    healthy_images, train_ratio, val_ratio, test_ratio)
  cancer_train, cancer_val, cancer_test = _make_dataset_splits(
    cancer_images, train_ratio, val_ratio, test_ratio)

  print("Healthy Patient Train/Val/Test: {}/{}/{}".format(
    len(healthy_train), len(healthy_val), len(healthy_test)))
  print("Cancer Patient Train/Val/Test: {}/{}/{}".format(
    len(cancer_train), len(cancer_val), len(cancer_test)))

  train_data_dict = {**healthy_train, **cancer_train}

  train_patient_ids = list(train_data_dict.keys())
  train_files = []
  train_size = 0
  for _, entries in train_data_dict.items():
    train_size += len(entries)
    for f in entries:
      train_files.append(f[0])

  assert(len(train_files) == train_size)

  val_data_dict = {**healthy_val, **cancer_val}

  val_patient_ids = list(val_data_dict.keys())
  val_files = []
  val_size = 0
  for _, entries in val_data_dict.items():
    val_size += len(entries)
    for f in entries:
      val_files.append(f[0])

  assert(len(val_files) == val_size)

  test_data_dict = {**healthy_test, **cancer_test}

  test_patient_ids = list(test_data_dict.keys())
  test_files = []
  test_size = 0
  for _, entries in test_data_dict.items():
    test_size += len(entries)
    for f in entries:
      test_files.append(f[0])
  assert(len(test_files) == test_size)

  dataset_size = train_size + val_size + test_size

  assert(dataset_size == num_healthy_images + num_cancer_images)

  print("Total dataset size: {}".format(dataset_size))

  print("Total Train/Val/Test Data: {}/{}/{}".format(
    train_size, val_size, test_size))

  data_dict = dict()
  data_dict[standard_fields.SplitNames.train] = train_data_dict
  data_dict[standard_fields.SplitNames.val] = val_data_dict
  data_dict[standard_fields.SplitNames.test] = test_data_dict

  result_dict = dict()
  result_dict[standard_fields.PickledDatasetInfo.data_dict] = data_dict
  result_dict[standard_fields.PickledDatasetInfo.patient_ids] = {
    standard_fields.SplitNames.train: train_patient_ids,
    standard_fields.SplitNames.val: val_patient_ids,
    standard_fields.SplitNames.test: test_patient_ids}
  result_dict[standard_fields.PickledDatasetInfo.file_names] = {
    standard_fields.SplitNames.train: train_files,
    standard_fields.SplitNames.val: val_files,
    standard_fields.SplitNames.test: test_files}
  result_dict[standard_fields.PickledDatasetInfo.patient_ratio] = patient_ratio
  result_dict[standard_fields.PickledDatasetInfo.seed] = seed

  with open(output_file, 'wb') as f:
    pickle.dump(result_dict, f)


if __name__ == '__main__':
  np.random.seed(args.seed)

  train_ratio = int(args.train_ratio)
  val_ratio = int(args.val_ratio)
  test_ratio = int(args.test_ratio)

  assert(train_ratio > 0)
  assert(val_ratio > 0)
  assert(test_ratio > 0)
  assert(train_ratio + val_ratio + test_ratio == 100)
  assert(os.path.isdir(args.dataset_dir))

  assign_patients(args.dataset_dir, train_ratio=train_ratio,
                  val_ratio=val_ratio, test_ratio=test_ratio,
                  only_cancer=args.only_cancer, seed=args.seed)
