#!/usr/bin/python3

import subprocess
import xml.etree.ElementTree
import logging
import sys
import pickle

import numpy as np

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

  all_gpu_mem = 0
  gpu_mems = []
  for gpu in gpus:
    total_mem_text = gpu.find('fb_memory_usage').find('total').text
    free_mem_text = gpu.find('fb_memory_usage').find('free').text
    assert total_mem_text[-3:] == 'MiB' and free_mem_text[-3:] == 'MiB'
    #total_mem = int(total_mem_text[:-4])
    free_mem = int(free_mem_text[:-4])
    #assert float(free_mem) / total_mem >= 0.85

    all_gpu_mem += free_mem
    gpu_mems.append(free_mem)

  res = []
  i = 0
  for gpu_mem in gpu_mems:
    pct = gpu_mem / all_gpu_mem
    res.append(('/device:GPU:{}'.format(i), pct))
    i += 1

  return res


def init_logger(folder=None, resume=None):
  # Logging Configuration
  formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] : %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')

  logging.root.setLevel(logging.DEBUG)

  sh = logging.StreamHandler(sys.stdout)
  sh.setLevel(logging.DEBUG)
  sh.setFormatter(formatter)
  logging.root.addHandler(sh)

  if folder:
    if resume:
      fh = logging.FileHandler(folder + '/log', mode='a')
    else:
      fh = logging.FileHandler(folder + '/log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)

  if resume and folder:
    logging.info("-------------------------")
    logging.info("Appending to existing log")
    logging.info("-------------------------")


def dense_to_one_hot(indices, cols):
  return np.eye(len(indices), cols)[indices]
