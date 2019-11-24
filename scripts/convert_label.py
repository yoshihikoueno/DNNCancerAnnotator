# built in
import os
import shutil
import pdb
import argparse

# external
import cv2
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


def extract_label(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    label = hsv[:, :, 2] > 180
    mask0 = hsv[:, :, 0] != 0
    mask1 = hsv[:, :, 1] != 0
    label = label & mask0 & mask1
    label = (label * 255).astype(np.uint8)
    return label

def crop_annotation_image(image):
    height, width, _ = image.shape
    output = image[:height // 2, :width // 2]
    return output

def crop_body(image):
    width_edge_right, width_edge_left = detect_end(image, axis=0)
    width_cropped = image[:, width_edge_right:width_edge_left]
    height_edge_top, height_edge_bottom = detect_end(width_cropped, axis=1)
    output = width_cropped[height_edge_top:height_edge_bottom]
    return output

def draw_center(image):
    height, width, _ = image.shape
    image[height // 2] = 255
    image[:, width // 2] = 255
    return image

def detect_end(image, axis=0):
    def for_each(array):
        if np.all(array < 5): return 1
        else: return 0

    image = image[:, :, 0]
    if axis != 0: image = image.T
    height, width = image.shape
    boundary_candidates = np.where(np.apply_along_axis(for_each, 0, image))[0]
    left_side = boundary_candidates < width // 2
    right_side = boundary_candidates > width // 2
    left_edge = np.max(boundary_candidates[left_side])
    right_edge = np.min(boundary_candidates[right_side])
    return left_edge, right_edge

def cutoff_edges(image, keepratio=0.80):
    height, width, _ = image.shape
    height_cutoff = (height - int(keepratio * height)) // 2
    width_cutoff = (width - int(keepratio * width)) // 2
    output = image[height_cutoff:-height_cutoff, width_cutoff:-width_cutoff]
    return output

def fill_label(label):
    background = np.zeros([*label.shape, 1], np.uint8)
    contour0, contour1 = np.where(label)
    points = np.stack([contour1, contour0], -1)
    cv2.fillPoly(background, [points], color=(255,))
    background = background[:, :, 0]
    return background

def add_label(image, label):
    assert image[:, :, 0].shape == label.shape
    output = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    output[:, :, 0] = output[:, :, 0] & ~label
    output[:, :, 1] = output[:, :, 1] & ~label
    output[:, :, -1] = (output[:, :, -1] & ~label) + label
    return output

def preview(image):
    cv2.imshow('preview', image)
    cv2.waitKey()
    return image

def save_label(target_dir):
    dirname = os.path.basename(target_dir)
    image = cv2.imread(os.path.join(target_dir, '{}.jpg'.format(dirname)))
    image = crop_annotation_image(image)
    image = crop_body(image)
    label = extract_label(image)
    filled_label = fill_label(label)
    image = add_label(image, filled_label)
    image = tf.image.resize_image_with_pad(image, 512, 512).numpy()
    cv2.imwrite(os.path.join(target_dir, 'label.jpg'), image)
    return

def test():
    image = cv2.imread('./15601715_15.jpg')
    image = crop_annotation_image(image)
    image = crop_body(image)
    label = extract_label(image)
    filled_label = fill_label(label)
    image = add_label(image, filled_label)
    image = tf.image.resize_image_with_pad(image, 512, 512).numpy()
    cv2.imwrite('label.jpg', image)
    return

def get_label_source_file_name(patient, exam, extension='jpg'):
    filename = f'{int(patient)}_{int(exam) % 100:02d}.{extension}'
    return filename

def process_patient_dir(patient_dir):
    assert os.path.exists(patient_dir), f'failed to find: {patient_dir}'

    print('Processing {}'.format(patient_dir))
    label_source = os.path.join(patient_dir, f'{os.path.basename(patient_dir)}.jpg')

    if not os.path.exists(label_source):
        print(f'Not found: {label_source}')
        return

    image = cv2.imread(label_source)
    image = crop_annotation_image(image)
    image = crop_body(image)
    label = extract_label(image)
    filled_label = fill_label(label)
    image = add_label(image, filled_label)
    image = tf.image.resize_image_with_pad(image, 512, 512).numpy()
    cv2.imwrite(os.path.join(patient_dir, 'label.jpg'), image)
    return

def process_patient_dir2(patient_dir):
    assert os.path.exists(patient_dir)

    exams = os.listdir(patient_dir)
    exam_dirs = list(filter(os.path.isdir, map(lambda x: os.path.join(patient_dir, x), exams)))

    for exam_dir in exam_dirs:
        print('Processing {}'.format(exam_dir))
        image = cv2.imread('./15601715_15.jpg')
        image = crop_annotation_image(image)
        image = crop_body(image)
        label = extract_label(image)
        filled_label = fill_label(label)
        image = add_label(image, filled_label)
        image = tf.image.resize_image_with_pad(image, 512, 512).numpy()
        cv2.imwrite('label.jpg', image)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    target_root = args.target

    for target_dir in os.listdir(target_root):
        print(target_dir)
        target_path = os.path.join(target_root, target_dir)
        if not os.path.isdir(target_path): continue
        process_patient_dir(target_path)
