from tensorflow/tensorflow:2.3.0rc1-gpu
run apt update
run apt install -y git python3-pip libsm6 libxext6 libxrender-dev
run git clone --single-branch --branch master https://github.com/yoshihikoueno/DNNCancerAnnotator
workdir DNNCancerAnnotator
run pip3 install -r requirements.txt
cmd NCCL_DEBUG=WARN NCCL_SHM_DISABLE=1 python3 -m annotator train\
    --config configs/unet.yaml\
    --save_path /kw_resources/results/annotation/db3/temp\
    --data_path\
        /kw_resources/datasets/projects/annotation/db3/train/cancer.tfrecords\
        /kw_resources/datasets/projects/annotation/db3/train/healthy.tfrecords\
    --max_steps 200000\
    --save_freq 500\
    --val_data_path\
        /kw_resources/datasets/projects/annotation/db3/val/cancer.tfrecords\
        /kw_resources/datasets/projects/annotation/db3/val/healthy.tfrecords\
    --validate\
    --visualize
