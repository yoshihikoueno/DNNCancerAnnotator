from tensorflow/tensorflow:2.1.1
run git clone --single-branch --brach restructure https://github.com/yoshihikoueno/DNNCancerAnnotator
workdir DNNCancerAnnotator
run pip3 install -r requirements.txt
cmd python3 -m annotator train\
    --config configs/unet.yaml\
    --save_path /kw_resources/results/annotation/db3/temp\
    --data_path data/train\
    --max_steps 20000\
    --save_freq 50
