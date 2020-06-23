from tensorflow/tensorflow:2.1.1
run apt update
run apt install -y git python3-pip libsm6 libxext6 libxrender-dev
run git clone --single-branch --branch restructure https://github.com/yoshihikoueno/DNNCancerAnnotator
workdir DNNCancerAnnotator
run pip3 install -r requirements.txt
cmd python3 -m annotator train\
    --config configs/unet.yaml\
    --save_path /kw_resources/results/annotation/db3/temp\
    --data_path /kw_resources/datasets/projects/annotation/db3/train\
    --max_steps 20000\
    --save_freq 50
