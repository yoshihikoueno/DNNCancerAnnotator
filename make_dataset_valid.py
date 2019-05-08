import os

from PIL import Image

dataset_dir = '/mnt/dataset/patrick/datasets/prostate_images3'

cancer_folder = os.path.join(dataset_dir, 'cancer_cases')
cancer_annotations = os.path.join(dataset_dir, 'cancer_annotations')

# Make sure cancer images and cancer annotations are same size
for patient_folder in os.listdir(cancer_folder):
  patient_folder = os.path.join(cancer_folder, patient_folder)
  for exam_folder in os.listdir(patient_folder):
    exam_folder = os.path.join(patient_folder, exam_folder)
    for image_file in os.listdir(exam_folder):
      image_file = os.path.join(exam_folder, image_file)
      annotation_file = os.path.join(
        cancer_annotations, os.path.basename(patient_folder),
        os.path.basename(exam_folder),
        os.path.basename(image_file).split('.')[0] + '.png')
      assert(os.path.exists(annotation_file))

      with open(image_file, 'rb') as f:
        image_pil = Image.open(image_file)
        image_size = image_pil.size

      # Make sure it is square
      assert(image_size[0] == image_size[1])

      with open(annotation_file, 'rb') as f:
        annotation_pil = Image.open(annotation_file)
        annotation_size = annotation_pil.size

      # Make sure it is square
      assert(annotation_size[0] == annotation_size[1])

      if image_size != annotation_size:
        print("Resizing annotation {}".format(annotation_file))
        annotation_pil = annotation_pil.resize(image_size,
                                               resample=Image.BICUBIC)
        annotation_pil.save(annotation_file)
