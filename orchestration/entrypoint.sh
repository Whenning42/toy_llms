set -e

# Set up private machine key
echo "$MACHINE_PRIVATE_KEY" > /root/.ssh/machine_key
chmod 600 /root/.ssh/machine_key

# We don't need to load our repo, since dstack does that for us.

rm -rf $MACHINE_OUT_DIR
mkdir -p $MACHINE_OUT_DIR
nvidia-smi --query-gpu=gpu_name --format=csv > $MACHINE_OUT_DIR/nvidia-smi.txt
python3 -c "import torch; print(torch.cuda.is_available())" > $MACHINE_OUT_DIR/torch_cuda.txt
python3 orchestration/job.py > $MACHINE_OUT_DIR/job_out.txt

# Copy the output back to the local machine
scp -rp -i /root/.ssh/machine_key -o "StrictHostKeyChecking accept-new" -P $LOCAL_PORT $MACHINE_OUT_DIR $LOCAL_USER@$LOCAL_HOST:$LOCAL_OUT_DIR
