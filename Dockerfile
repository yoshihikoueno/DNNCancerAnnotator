from tensorflow/tensorflow:1.13.1-gpu-py3
run install_proto.sh
run git clone --single-branch --branch dev https://github.com/yoshihikoueno/DNNCancerAnnotator
workdir DNNCancerAnnotator
run protc --python_out=./ ./protos/*.prot
cmd python3 -m runs.train\
    --num_train_steps 2000\
    --pipeline_config_file default.config\
    --result_dir /kw_resources/results
