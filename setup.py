from distutils.core import setup

with open('./requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='DNNCancerAnnotator',
    description='AI to predict prostate cancer annotations using DNN (Deep Neural Network)',
    packages=requirements,
)
