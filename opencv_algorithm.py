import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageEnhance

def preprocess_image(img):
  # Make the image quadratic
  if img.shape[0] > img.shape[1]:
    # Vertical crop
    crop_size = img.shape[0] - img.shape[1]
    img = img[int(np.ceil(crop_size / 2.0)):img.shape[0] - int(np.floor(
      crop_size / 2.0)),:]
  elif img.shape[1] > img.shape[0]:
    # Horizontal crop
    crop_size = img.shape[1] - img.shape[0]
    img = img[:,int(np.ceil(crop_size / 2.0)):img.shape[1] - int(np.floor(
      crop_size / 2.0))]

  assert img.shape[0] == img.shape[1]

  # Resize to a common size
  common_size = 512
  img = cv2.resize(img, (common_size, common_size),
                   interpolation=cv2.INTER_LANCZOS4)

  final_size = 256
  crop_size = img.shape[0] - final_size
  img = img[int(np.ceil(crop_size / 2.0)):img.shape[0] - int(np.floor(
    crop_size / 2.0))
  ,int(np.ceil(crop_size / 2.0)):img.shape[1] - int(np.floor(crop_size /
                                                               2.0))]
  assert img.shape[0] == img.shape[1] == final_size

  return img

def run():
  image_folder_path = '/home/patrick/projects/NetNextGen/opencv_resources/opencv_test_images'
  gt_image_path = '/home/patrick/projects/NetNextGen/opencv_resources/prostate_gt.jpg'
  image_names = os.listdir(image_folder_path)
  image_paths = [os.path.join(image_folder_path, image_name) for image_name
                 in image_names]

  for image_path in image_paths[4:]:
    original_image = cv2.imread(image_path)
    original_image = preprocess_image(original_image)
    original_image2 = np.copy(original_image)
    #cv2.imshow("Original", original_image)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    contrast = np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(2.5))
    #cv2.imshow("Contrast", contrast)
    
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    #cv2.imshow("Blur", blur)
    rt, threshold = cv2.threshold(blur, 0, 255, type=cv2.THRESH_BINARY_INV +
                                             cv2.THRESH_OTSU)

    #cv2.imshow("Threshold", threshold)

    circles = cv2.HoughCircles(contrast, cv2.HOUGH_GRADIENT, 1.08, 200, param1=200,
                               param2=40, minRadius=20, maxRadius=60)
    circles = np.uint16(np.around(circles))
    
    areas = np.ones_like(original_image, dtype=np.uint8)
    
    for c in circles[0]:
      cv2.circle(areas, (c[0], c[1]), c[2]+15, (2, 2, 2), -1)
      cv2.circle(areas, (c[0], c[1]), c[2]+15, (0, 0, 0), 15)
    cv2.imshow("Areas", areas * 50)
    print(areas.dtype)

    areas = areas[:,:,0]
    
    areas = np.int32(areas)
    markers = cv2.watershed(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR), areas)
    original_image[markers == -1] = [0, 0, 255]

    cv2.imshow("Watershed", original_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #img = cv2.GaussianBlur(img, (5, 5), 0)
    #cv2.imshow("Blur", img)

    #ret, img = cv2.threshold(img, 255, 255, type=cv2.THRESH_BINARY_INV +
    #                                         cv2.THRESH_OTSU)
    #cv2.imshow("Threshold", img)

    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8),
    #                       iterations=1)
    #cv2.imshow("Close", img)

    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    #cv2.imshow("Close2", img)

    #img = np.uint8(np.clip(cv2.Laplacian(img, cv2.CV_64F), 0,
    #                       1) * 255)
    #cv2.imshow("Laplacian", img)

    #img = cv2.GaussianBlur(img, (15, 15), 0)
    #cv2.imshow("Blur 2", img)


    #img = cv2.Canny(original_image, 200, 300)
    #cv2.imshow("Edges before blur", img)

    #img = cv2.GaussianBlur(original_image, (7, 7), 0)

    #ret, thresh = cv2.threshold(blur, 100, 255,
    #                            cv2.THRESH_BINARY)
    #cv2.imshow("After Blur", img)

    #kernel = np.ones((2, 2), np.uint8)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    #cv2.imshow("After opening", opening)

    #sure_bg = cv2.dilate(opening, kernel, iterations=2)
    #cv2.imshow("Sure background", sure_bg)

    #kernel = np.ones((3, 3), np.uint8)
    #erosion = cv2.erode(opening, kernel, iterations=2)
    #ret, sure_fg = cv2.threshold(erosion, 0.7 * erosion.max(),
    #                               255, 0)
    #cv2.imshow("Sure foreground", sure_fg)

    # Finding unknown region
    #sure_fg = np.uint8(sure_fg)
    #unknown = cv2.subtract(sure_bg, sure_fg)
    #cv2.imshow("Unknown", unknown)

    # Marker labelling
    #ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    #markers = markers + 1
    # Now, mark the region of unknown with zero
    #markers[unknown == 255] = 0

    #img = cv2.Canny(img, 100, 200)
    #cv2.imshow("Edges", img)

    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8),
    #                       iterations=2)
    #cv2.imshow("Open", img)

    #im2, contours, hierarchy = cv2.findContours(255 - img, cv2.RETR_TREE,
    #                                 cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("Contours514", im2)
    #im3 = np.copy(im2)
    #im2 = cv2.cvtColor(im2,cv2.COLOR_GRAY2BGR)
    #im3 = cv2.cvtColor(im3, cv2.COLOR_GRAY2BGR)

    #cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)
    #cv2.imshow("Contours", im2)

    #def fun(x):
    #  return cv2.contourArea(x)>10

    #contours2 = list(filter(fun,contours))




    #cv2.drawContours(im3, contours2, -1, (0,0,255), 3)
    #cv2.imshow("Contours2", im3)

    #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.08, 200, param1=200,
    #                           param2=40, minRadius=40, maxRadius=110)
    #circles = np.uint16(np.around(circles))

    #for c in circles[0, :]:
    #  cv2.circle(original_image2, (c[0], c[1]), c[2], (0, 0, 255), 2)
    #cv2.imshow("Second Circles", original_image2)

    #m = img.astype(np.int32)
    #m[int(m.shape[0]/2), int(m.shape[1]/2)] = 0
    #markers = cv2.watershed(original_image, m)
    #original_image[markers == -1] = [0, 0, 255]

    #cv2.imshow("Final", original_image)

  exit(1)




  # Preprocess images
  # for image_path in image_paths:
  #   img = cv2.imread(image_path)
  #   img = preprocess_image(img)
  #
  #   # Perform watershed segmentation
  #   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #
  #   for thresh_name, thresh_method in threshold_methods:
  #     ret, thresh = cv2.threshold(gray, 0, 255, thresh_method)
  #     cv2.imshow(thresh_name, thresh)
  #     cv2.waitKey(0)
  #   exit(1)
  #
  #   # Use Otsu binarization
  #   #ret, thresh = cv2.threshold(gray, 0, 255,
  #   #                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  #   #plt.imshow(thresh), plt.show()
  #
  #   # noise removal
  #   kernel = np.ones((3, 3), np.uint8)
  #   opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
  #   plt.imshow(opening), plt.show()
  #
  #   # sure background area
  #   sure_bg = cv2.dilate(opening, kernel, iterations=3)
  #   plt.imshow(sure_bg), plt.show()
  #
  #   # Finding sure foreground area
  #   #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
  #   kernel = np.ones((5, 5), np.uint8)
  #   erosion = cv2.erode(opening, kernel, iterations=1)
  #   ret, sure_fg = cv2.threshold(erosion, 0.7 * erosion.max(),
  #                                255, 0)
  #   plt.imshow(sure_fg), plt.show()
  #
  #   # Finding unknown region
  #   sure_fg = np.uint8(sure_fg)
  #   unknown = cv2.subtract(sure_bg, sure_fg)
  #   plt.imshow(unknown), plt.show()
  #
  #   # Marker labelling
  #   ret, markers = cv2.connectedComponents(sure_fg)
  #
  #   # Add one to all labels so that sure background is not 0, but 1
  #   markers = markers + 1
  #   # Now, mark the region of unknown with zero
  #   markers[unknown == 255] = 0
  #   markers = cv2.watershed(img, markers)
  #   img[markers == -1] = [255, 0, 0]
  #
  #   plt.imshow(img), plt.show()
  #   exit(1)
  # exit(1)



  # Extract features
  gt_img = cv2.imread(gt_image_path)
  gt_img = cv2.resize(gt_img, (256, 256),
             interpolation=cv2.INTER_LANCZOS4)
  gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

  # SIFT detector
  # detector = cv2.xfeatures2d.SIFT_create()
  # ORB detector
  detector = cv2.ORB_create()
  gt_kp, gt_des = detector.detectAndCompute(gt_img, None)

  per_image_features = []
  for image_path in image_paths:
    img = cv2.imread(image_path)
    img = preprocess_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = detector.detectAndCompute(img, None)
    per_image_features.append((img, kp, des))

    #img = cv2.drawKeypoints(img, kp, None,
    #                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #cv2.imwrite("/home/patrick/Desktop/tmp.jpg", img)

  # SIFT matcher
  #matcher = cv2.BFMatcher(cv2.NORM_L2, False)
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING, True)

  matches = matcher.match(gt_des, per_image_features[0][2])

  # Sort them in the order of their distance.
  matches = sorted(matches, key=lambda x: x.distance)
  # Draw first 20 matches
  img3 = cv2.drawMatches(gt_img, gt_kp,
                         per_image_features[2][0],
                         per_image_features[2][1], matches[:20],
                         outImg=None, flags=2)

  plt.imshow(img3), plt.show()
if __name__ == '__main__':
  run()
