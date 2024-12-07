# This will be the "main" for the job you run on the cloud machine.

nvidia-smi --query-gpu=gpu_name --format=csv > $MACHINE_OUT_DIR/nvidia-smi.txt
python3 -c "import torch; print(torch.cuda.is_available())" > $MACHINE_OUT_DIR/torch_cuda.txt
echo "You've successfully run a job on a cloud machine!" > $MACHINE_OUT_DIR/job_out.txt
