set -e

# Source the scp config env file
echo "$SCP_CONFIG" > .scp_config
set -o allexport
source .scp_config

# Set up private machine key
echo "$MACHINE_PRIVATE_KEY" > /root/.ssh/machine_key
chmod 600 /root/.ssh/machine_key

# Set up the job's output directory
rm -rf $MACHINE_OUT_DIR
mkdir -p $MACHINE_OUT_DIR

# Run the specified command
eval "$RUN_COMMAND"

# Copy the output back to the local machine
scp -rp -i /root/.ssh/machine_key -o "StrictHostKeyChecking accept-new" -P $LOCAL_PORT $MACHINE_OUT_DIR $LOCAL_USER@$LOCAL_HOST:$LOCAL_OUT_DIR
