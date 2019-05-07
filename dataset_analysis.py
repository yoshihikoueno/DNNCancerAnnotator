import os

from PIL import Image

# Information to extract
# From full sequence dataset
# - avg / min / max number of slices per patient
# - Image sizes
# - Patients not present in previous dataset + Num
# - Num total patients in new / old
# - Num total images in new / old

folder = '/mnt/dataset/patrick/datasets'
old_dataset_folder = os.path.join(folder, 'prostate_images3')
old_dataset_healthy_folder = os.path.join(old_dataset_folder, 'healthy_cases')
old_dataset_cancer_folder = os.path.join(old_dataset_folder, 'cancer_cases')
old_dataset_cancer_annotations_folder = os.path.join(old_dataset_folder,
                                                     'cancer_annotations')
new_dataset_folder = os.path.join(folder, 'prostate_images_full_sequence')

result_dict = dict()
result_dict['old_num_images'] = 0
result_dict['new_num_images'] = 0
result_dict['old_num_patients_healthy'] = 0
result_dict['old_num_patients_cancer'] = 0
result_dict['label_min_slice_index'] = (999, None)
result_dict['label_max_slice_index'] = (0, None)
result_dict['label_index_max_distance'] = (0, None)
result_dict['new_num_patients'] = 0

result_dict['old_patient_list'] = []
result_dict['new_patient_list'] = []

result_dict['old_patients_with_holes'] = []
result_dict['new_patients_with_holes'] = []


result_dict['image_size_tuple_to_num'] = dict()
result_dict['min_num_slices'] = (999, None)
result_dict['max_num_slices'] = (0, None)

result_dict['num_patients_with_min_32_slices'] = 0


def has_holes(slice_indices):
  i0 = slice_indices[0]

  for i in slice_indices[1:]:
    if i - 1 != i0:
      return True
    i0 = i

  return False


def walk_dir(directory, is_old, id_prefix, result_dict):
  patient_folders = os.listdir(directory)

  for patient_folder in patient_folders:
    patient_folder = os.path.join(directory, patient_folder)
    exam_folders = os.listdir(patient_folder)
    patient_id = id_prefix + os.path.basename(patient_folder)

    num_files = 0
    files = dict()

    for exam_folder in exam_folders:
      exam_folder = os.path.join(patient_folder, exam_folder)
      exam_files = os.listdir(exam_folder)
      slice_indices = []
      if len(exam_files) == 0:
        print(exam_folder)
        assert(len(exam_files) > 0)
      files[os.path.basename(exam_folder)] = exam_files
      num_files += len(exam_files)

      if not is_old:
        if len(exam_files) < result_dict['min_num_slices'][0]:
          result_dict['min_num_slices'] = (len(exam_files), patient_id)
        if len(exam_files) > result_dict['max_num_slices'][0]:
          result_dict['max_num_slices'] = (len(exam_files), patient_id)

      for f in exam_files:
        assert(len(f.split('_')) == 1 if is_old else len(f.split('_')) == 2)
        (slice_indices.append(int(f.split('.')[0])) if is_old else
         slice_indices.append(int(f.split('.')[0].split('_')[1])))
        with open(os.path.join(exam_folder, f), 'rb') as fp:
          f_pil = Image.open(fp)
          size = f_pil.size
        if size not in result_dict['image_size_tuple_to_num']:
          result_dict['image_size_tuple_to_num'][size] = 0
          result_dict['image_size_tuple_to_num'][size] += 1

      if id_prefix == 'c':
        if min(slice_indices) < result_dict['label_min_slice_index'][0]:
          result_dict['label_min_slice_index'] = (
            min(slice_indices), patient_id)
        if max(slice_indices) > result_dict['label_max_slice_index'][0]:
          result_dict['label_max_slice_index'] = (
            max(slice_indices), patient_id)

        label_distance = max(slice_indices) - min(slice_indices) + 1

        if label_distance > result_dict['label_index_max_distance'][0]:
          result_dict['label_index_max_distance'] = (
            label_distance, patient_id)

      slice_indices.sort()
      if has_holes(slice_indices):
        if is_old:
          if patient_id not in result_dict['old_patients_with_holes']:
            result_dict['old_patients_with_holes'].append(patient_id)
        else:
          if patient_id not in result_dict['new_patients_with_holes']:
            result_dict['new_patients_with_holes'].append(patient_id)

    if is_old:
      assert(patient_id not in result_dict['old_patient_list'])
      result_dict['old_patient_list'].append(patient_id)
      result_dict['old_num_images'] += num_files
      if id_prefix == 'h':
          result_dict['old_num_patients_healthy'] += 1
      else:
          result_dict['old_num_patients_cancer'] += 1
    else:
      assert(patient_id not in result_dict['new_patient_list'])
      result_dict['new_patient_list'].append(patient_id)
      result_dict['new_num_patients'] += 1
      result_dict['new_num_images'] += num_files


walk_dir(old_dataset_healthy_folder, is_old=True, id_prefix='h',
         result_dict=result_dict)
walk_dir(old_dataset_cancer_folder, is_old=True, id_prefix='c',
         result_dict=result_dict)
walk_dir(new_dataset_folder, is_old=False, id_prefix='',
         result_dict=result_dict)

result_dict['unassigned_patient_ids'] = []
for patient_id in result_dict['new_patient_list']:
  c_patient_id = 'c' + patient_id
  h_patient_id = 'h' + patient_id

  if (c_patient_id not in result_dict['old_patient_list']
      and h_patient_id not in result_dict['old_patient_list']):
    result_dict['unassigned_patient_ids'].append(patient_id)

print(result_dict)
