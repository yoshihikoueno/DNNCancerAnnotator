'''
extract data from screenshot provided by Dr.Matuoka@TMDU
'''

# buit-in
import pdb
import os
import argparse
import itertools
from glob import glob
from multiprocessing.pool import Pool
import multiprocessing as mp
import functools

# external
import cv2
import numpy as np
from scipy import signal
import dsargparse
from tqdm import tqdm
import p_tqdm


def get_orthogonal_detector(size=200):
    '''
    generates conv filter to detect orthogonal
    corners
    '''
    filter_ = np.zeros([size] * 2)
    filter_[0, :] = 1
    filter_[:, 0] = 1
    return filter_


def find_top_left_fallback(gray):
    '''fallback method to find top left corner
    conv-based method should be used instead of this method
    when it's possible.
    '''
    def find_top(start=120):
        current = start
        while np.sum(gray[current, 100:700]) != 0:
            current += 1
        return current

    def find_left(start=120):
        current = start
        while np.sum(gray[250:800, current]) != 0:
            current -= 1
        return current

    x, y = find_top() + 3, find_left() - 75
    return x, y


def detect_internals(
    collective_img,
    num_internals=6,
    separator_value=200,
    conv_filter_size=20,
    box_size=(708, 850),
    nboxes_horizontal=3,
    debug_output=None,
):
    '''
    detect starting points of 6 internal images
    contained in `collective_img`.

    Args:
        collective_img: img shape:W,H,D

    Returns:
        list of boxes (startx, starty, endx, endy)
    '''
    gray = collective_img[:, :, 0]
    filtered = gray == separator_value
    conv_filter = get_orthogonal_detector(conv_filter_size)

    conv_result = signal.convolve2d(filtered, np.flip(conv_filter), 'valid')
    corners = conv_result == (conv_filter_size * 2 - 1)
    xs, ys = np.where(corners)
    if len(xs) > 0:
        target_idx = np.argmin(xs)
        x, y = xs[target_idx], ys[target_idx]
        # make sure (x,y) is pointing to the top-left corner
        while x > 200: x -= box_size[0]
        while y > 60: y -= box_size[1]
    else:
        x, y = find_top_left_fallback(gray)
        if x < 0 or y < 0: raise ValueError('Failed to detect corners')

    anchor = x, y
    first_anchor = anchor
    boxes = []
    for i in range(num_internals):
        box_end = anchor[0] + box_size[0], anchor[1] + box_size[1]
        boxes.append((anchor[0] + 1, anchor[1] + 1, box_end[0], box_end[1]))
        if (i + 1) % nboxes_horizontal == 0:
            anchor = first_anchor[0] + box_size[0], first_anchor[1]
        else:
            anchor = anchor[0], anchor[1] + box_size[1]

    if debug_output is not None:
        cv2.imwrite(os.path.join(debug_output, 'gray.png'), gray)
        cv2.imwrite(os.path.join(debug_output, 'conv_filter.png'), conv_filter * 255)
        cv2.imwrite(os.path.join(debug_output, 'conv_result.png'), conv_result * 255)
        # pdb.set_trace()
        # cv2.imwrite(os.path.join(debug_output, 'corners.png'), corners * 255.0)
    return boxes


def extract_images(collective_img, boxes):
    '''
    for a given collective image, this func will extract
    internal images according to boxes
    '''
    imgs = [
        collective_img[startx:endx, starty:endy]
        for startx, starty, endx, endy in boxes
    ]
    return imgs

def get_monochrome_pixels(img):
    '''
    get monochrome pixels
    '''
    monochrome = np.logical_and(np.equal(img[:, :, 0], img[:, :, 1]), np.equal(img[:, :, 1], img[:, :, 2]))
    return monochrome


def get_center_mask(size, radius=130, dtype=np.uint8):
    '''
    return a mask to filter outside the central circle
    '''
    assert isinstance(size, (list, tuple))
    input_dims = len(size)
    assert input_dims == 3

    mask = np.zeros(size, dtype=dtype)
    cv2.circle(mask, (size[1] // 2, size[0] // 2), radius, color=255, thickness=-1)

    return mask


def label_exists(label_img, minLineLength=3):
    '''checks if a label image contains annotations'''
    monochrome = get_monochrome_pixels(label_img)
    color = np.logical_not(monochrome)
    color = (np.expand_dims(color, -1) * 255).astype(np.uint8)

    result = cv2.HoughLinesP(color, 1, np.pi / 180 / 2, 100, minLineLength=minLineLength)
    return result is not None


def extract_label(
    label_img,
    line_eraser_thickness=3,
    minLineLength=3,
    debug_output=None,
):
    '''
    detect label and return filled label image
    '''
    monochrome = get_monochrome_pixels(label_img)
    color = np.logical_not(monochrome)
    color = (np.expand_dims(color, -1) * 255).astype(np.uint8)

    lines = cv2.HoughLinesP(color, 1, np.pi / 180 / 2, 100, minLineLength=minLineLength)[0]
    color_nolines = color.copy()
    for x0, y0, x1, y1 in lines:
        cv2.line(color_nolines, (x0, y0), (x1, y1), 0, line_eraser_thickness)

    center_masked = np.logical_and(get_center_mask(color_nolines.shape), color_nolines).astype(np.uint8) * 255
    closed = cv2.morphologyEx(center_masked, cv2.MORPH_CLOSE, np.ones([7] * 2, np.uint8))
    ctrs, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    label = np.zeros(color.shape, dtype=np.uint8)
    cv2.fillPoly(label, ctrs, 255)

    if debug_output is not None:
        cv2.imwrite(os.path.join(debug_output, 'extract_label_input.png'), label_img)
        cv2.imwrite(os.path.join(debug_output, 'extract_label_gray.png'), monochrome)
        cv2.imwrite(os.path.join(debug_output, 'color.png'), color)
        cv2.imwrite(os.path.join(debug_output, 'color_nolines.png'), color_nolines)
        cv2.imwrite(os.path.join(debug_output, 'center_masked.png'), center_masked)
        cv2.imwrite(os.path.join(debug_output, 'closed.png'), closed)
    return label


def save_output(output, result):
    os.makedirs(output, exist_ok=True)
    for tag, img in result.items():
        cv2.imwrite(os.path.join(output, f'{tag}.png'), img)
    return


def extract(
    path,
    output,
    include_label=False,
    debug_output=None,
    include_label_comparison=False,
):
    '''
    extract data

    Args:
        path: input image path
        output: output image output directory
        include_label: should label image also generated
        debug_output: debug image output directory
            default: None (disabled)
        include_label_comparison: should this func also export
            an image combining annotation image and segmentation image
    '''
    if debug_output is not None: os.makedirs(debug_output, exist_ok=True)

    collective_img = cv2.imread(path)
    assert collective_img is not None, f'failed to load {path}'
    try: boxes = detect_internals(collective_img, debug_output=debug_output)
    except ValueError: raise ValueError(f'Failed to detect corners: {path}')
    imgs = extract_images(collective_img, boxes)

    result = {
        'DCEE': imgs[1], 'DCEL': imgs[2],
        'DWI': imgs[3], 'ADC': imgs[4], 'TRA': imgs[5],
    }

    if include_label:
        assert label_exists(imgs[0]), f'{path} doen\'t seem to have a label'
        label = imgs[0]
        label = extract_label(label, debug_output=debug_output)
        result['label'] = label

    if include_label_comparison:
        assert include_label, 'label must be included to include label_comparison'
        result['label_comparison'] = np.concatenate([np.expand_dims(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY), axis=-1), label], axis=1)
    else: assert not label_exists(imgs[0])

    if output is not None: save_output(output, result)
    return result


def extract_all(path, dry=False, debug=False):
    '''
    extract indivisual images (TRA, ADC, etc...) from the screenshots
    under the specified directory.

    Args:
        path: directory which contains screenshots.
            under the path, it should be structured as below:
            -- path -- healthy -- patientID -- examID -- <sliceID>.png
                    |- cancer -- patientID -- examID -- <sliceID>.png
        dry: should do the dry run.
            this will make no changes to the disk.
            useful to make sure that it doesn't fail
        debug: should also output debug image
    '''
    assert os.path.exists(path)
    healthy_path = os.path.join(path, 'healthy')
    cancer_path = os.path.join(path, 'cancer')
    assert os.path.exists(healthy_path) and os.path.exists(cancer_path)

    tasks = {'slice': [], 'exam': [], 'include_label': [], 'debug': [], 'dry': []}

    # process healthy cases
    for exam, slices in tqdm(list_exams(healthy_path).items(), desc='healthy cases', leave=False):
        for slice_ in slices:
            tasks['slice'].append(slice_)
            tasks['exam'].append(exam)
            tasks['include_label'].append(False)
            tasks['dry'].append(dry)
            tasks['debug'].append(False)

    # process cancer cases
    for exam, slices in tqdm(list_exams(cancer_path).items(), desc='cancer cases', leave=False):
        for slice_ in slices:
            tasks['slice'].append(slice_)
            tasks['exam'].append(exam)
            tasks['include_label'].append(True)
            tasks['dry'].append(dry)
            tasks['debug'].append(debug)

    p_tqdm.p_map(
        process_slice,
        tasks['slice'], tasks['exam'], tasks['dry'], tasks['include_label'], tasks['debug'],
    )
    return

def process_slice(slice_, exam, dry, include_label, debug):
    results = extract(os.path.join(exam, slice_), None, include_label=include_label, include_label_comparison=debug)
    for kind, img in results.items():
        kind_dir = os.path.join(exam, kind)
        if not dry:
            if not os.path.exists(kind_dir): os.makedirs(kind_dir, exist_ok=True)
            cv2.imwrite(os.path.join(kind_dir, slice_), img)
    return


def list_exams(path, extension='png'):
    if path[-1] == os.path.sep: path = path[:-1]
    exams = {
        exam: sorted(list(filter(lambda x: os.path.splitext(x)[1][1:] == extension, os.listdir(exam))))
        for exam in glob(os.path.join(path, '*', '*'))
        if list(filter(lambda x: os.path.splitext(x)[1][1:] == extension, os.listdir(exam)))
    }
    return exams


def main():
    parser = dsargparse.ArgumentParser(main=main)
    subparser = parser.add_subparsers()
    subparser.add_parser(extract, add_arguments_auto=True)
    return parser.parse_and_run()


if __name__ == '__main__':
    main()
