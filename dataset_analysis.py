import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cc3d

folder = '/mnt/dataset/patrick/datasets'
old_dataset_folder = os.path.join(folder, 'prostate_images4')
old_dataset_healthy_folder = os.path.join(old_dataset_folder, 'healthy_cases')
old_dataset_cancer_folder = os.path.join(old_dataset_folder, 'cancer_cases')
old_dataset_cancer_annotations_folder = os.path.join(old_dataset_folder,
                                                     'cancer_annotations')

result_dict = dict()
result_dict['old_num_images'] = 0
result_dict['old_num_patients_healthy'] = 0
result_dict['old_num_patients_cancer'] = 0
result_dict['label_min_slice_index'] = (999, None)
result_dict['label_max_slice_index'] = (0, None)
result_dict['label_index_max_distance'] = (0, None)

result_dict['old_patient_list'] = []

result_dict['old_patients_with_holes'] = []


result_dict['image_size_tuple_to_num'] = dict()
result_dict['min_num_slices'] = (999, None)
result_dict['max_num_slices'] = (0, None)

result_dict['num_patients_exams_min_16_slices'] = 0
result_dict['num_patients_exams_min_24_slices'] = 0
result_dict['num_patients_exams_min_32_slices'] = 0
result_dict['num_total_patient_exams'] = 0
result_dict['num_healthy_patient_exams'] = 0

result_dict['num_slices_with_lesion'] = 0

num_slices_histogram = []
num_lesions = []
num_lesions_patients = []

def has_holes(slice_indices):
  i0 = slice_indices[0]

  for i in slice_indices[1:]:
    if i - 1 != i0:
      return True
    i0 = i

  return False


def has_lesion(image):
  np_img = np.array(image).astype(np.int32)

  bool_mask = np.greater(np_img[:, :, 0] - np_img[:, :, 1], 200)

  return np.any(bool_mask)


def count_lesions(annotation_image):
  np_img = np.array(annotation_image).astype(np.int32)

  annotation_mask = np.greater(np_img[:, :, 0] - np_img[:, :, 1], 200)

  components = cc3d.connected_components(annotation_mask)

  return np.max(components)


def walk_dir(directory, is_old, id_prefix, result_dict, annotation_dir=None):
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

      # Histogram
      num_slices_histogram.append(len(exam_files))

      result_dict['num_total_patient_exams'] += 1
      if id_prefix == 'h':
        result_dict['num_healthy_patient_exams'] += 1
      if len(exam_files) >= 16:
        result_dict['num_patients_exams_min_16_slices'] += 1
      if len(exam_files) >= 24:
        result_dict['num_patients_exams_min_24_slices'] += 1
      if len(exam_files) >= 32:
        result_dict['num_patients_exams_min_32_slices'] += 1

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
        if annotation_dir:
          annotation_path = os.path.join(
            annotation_dir, os.path.basename(patient_folder),
            os.path.basename(exam_folder), '{}.png'.format(f.split('.')[0]))
          with open(annotation_path, 'rb') as fp:
            f_pil = Image.open(fp)
            with_lesion = has_lesion(f_pil)
          if with_lesion:
            result_dict['num_slices_with_lesion'] += 1
            lesion_count = count_lesions(f_pil)
            assert(lesion_count > 0)
            num_lesions.append(lesion_count)
          else:
            num_lesions.append(0)
          num_lesions_patients.append('{}_{}'.format(patient_id,
                                                     slice_indices[-1]))
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
         result_dict=result_dict,
         annotation_dir=old_dataset_cancer_annotations_folder)

# Plot Histogram
#x = []
#y = np.unique(num_slices_histogram)
#for u_v in y:
#  x.append(np.where(np.array(num_slices_histogram) == u_v)[0])

#for i in range(len(x)):
#  x[i] = np.array(x[i]).size

#plt.bar(x, y, width=1.0, align='edge')
plt.hist(num_slices_histogram, bins=list(range(1, 80)), density=False,
         facecolor='b', alpha=0.75)
plt.xlabel('Number of slices')
plt.ylabel('Number of examinations')
plt.title('Histogram for the number of slices.')
plt.grid(True)
plt.savefig('slice_histogram.png', bbox_inches='tight')

print(result_dict)
mean = sum(num_slices_histogram) / len(num_slices_histogram)
print("Mean num slices: {}".format(mean))
variance = 0
for v in num_slices_histogram:
  variance += (v - mean)**2
variance /= (len(num_slices_histogram) - 1)
print("Variance num slices: {}".format(variance))
print('Mean num lesions: {}'.format(np.mean(num_lesions)))
print('Total number of lesions: {}'.format(np.sum(num_lesions)))
print('1 Lesion: {}'.format(np.sum(np.array(num_lesions) == 1)))
print('2 Lesions: {}'.format(np.sum(np.array(num_lesions) == 2)))
print('3 Lesions: {}'.format(np.sum(np.array(num_lesions) == 3)))
print(np.array(num_lesions_patients)[np.where(np.array(num_lesions) == 3)[0]])

print('4 Lesions: {}'.format(np.sum(np.array(num_lesions) == 4)))
print(np.array(num_lesions_patients)[np.where(np.array(num_lesions) == 4)[0]])

print('More Lesions: {}'.format(np.sum(np.array(num_lesions) > 4)))
print(np.array(num_lesions_patients)[np.where(np.array(num_lesions) > 4)[0]])

plt.hist(num_lesions, bins=list(range(0, 5)), density=False,
         facecolor='b', alpha=0.75)
plt.xlabel('Number of lesions')
plt.xticks([0, 1, 2, 3, 4])
plt.ylabel('Number of image slices')
plt.title('Histogram of the number of slices')
plt.grid(True)
plt.savefig('num_lesions_histogram.png', bbox_inches='tight')
