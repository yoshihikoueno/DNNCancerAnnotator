import io

import numpy as np
import tensorflow as tf
from PIL import Image

# desired_size = List with two elements describing height and width
def central_crop(img, desired_size):
  assert(len(img.get_shape().as_list()) == 4)
  assert(len(desired_size) == 2)
  desired_size = np.array(desired_size)
  img_size = np.array(img.get_shape().as_list()[1:3])

  assert((desired_size > img_size).sum() == 0)

  offsets = ((img_size - desired_size) / 2).astype(np.int32)

  cropped_img = tf.image.crop_to_bounding_box(img, offsets[0], offsets[1],
                                              desired_size[0], desired_size[1])

  return cropped_img


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = io.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string
