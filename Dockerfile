from tensorflow/tensorflow:1.13.1-gpu-py3
run apt update && apt install --no-install-recommends -y git libopencv-dev
run git clone --single-branch --branch dev https://github.com/yoshihikoueno/DNNCancerAnnotator
workdir DNNCancerAnnotator
run pip3 install -U pip
run pip3 install -r requirements.txt
run chmod +x ./install_proto.sh
run ./install_proto.sh
run protoc --python_out=./ ./protos/*.proto
arg date
run git pull
cmd python3 -m runs.train\
    --num_train_steps 2000\
    --pipeline_config_file default.config\
    --result_dir /kw_resources/results/annotation
