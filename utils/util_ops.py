#!/usr/bin/python3

import subprocess
import xml.etree.ElementTree
import logging
import sys
import multiprocessing
import os

import numpy as np
import tensorflow as tf


def get_cpu_count():
  return multiprocessing.cpu_count()


def get_devices():
  res = []
  try:
    subprocess.check_output(['which', 'nvidia-smi']).decode('utf-8')
  except subprocess.CalledProcessError:
    # Only CPU available
    res.append(('/device:CPU:0', 1))
    return res

  xml_output = subprocess.check_output(['nvidia-smi', '-q', '-x']).decode(
    'utf-8')

  e = xml.etree.ElementTree.fromstring(xml_output)

  gpus = e.findall('gpu')

  res = []
  for i, gpu in enumerate(gpus):
    total_mem_text = gpu.find('fb_memory_usage').find('total').text
    free_mem_text = gpu.find('fb_memory_usage').find('free').text
    assert total_mem_text[-3:] == 'MiB' and free_mem_text[-3:] == 'MiB'
    total_mem = int(total_mem_text[:-4])
    free_mem = int(free_mem_text[:-4])
    #assert float(free_mem) / total_mem >= 0.85

    res.append((i, float(free_mem) / total_mem))

  return res


def init_logger(folder=None, resume=None):
  # Logging Configuration
  formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] : %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')

  logging.getLogger().setLevel(logging.DEBUG)

  sh = logging.StreamHandler(sys.stdout)
  sh.setLevel(logging.DEBUG)
  sh.setFormatter(formatter)
  logging.getLogger().addHandler(sh)

  tf.logging.set_verbosity(tf.logging.DEBUG)
  logging.getLogger('tensorflow').handlers = []

  if folder:
    if resume:
      fh = logging.FileHandler(os.path.join(folder, 'log'), mode='a')
    else:
      fh = logging.FileHandler(os.path.join(folder, 'log'), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)

  if resume and folder:
    logging.info("-------------------------")
    logging.info("Appending to existing log")
    logging.info("-------------------------")


def dense_to_one_hot(indices, cols):
  return np.eye(len(indices), cols)[indices]
