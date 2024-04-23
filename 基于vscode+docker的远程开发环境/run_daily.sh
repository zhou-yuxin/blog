image_name=nvcr.io/nvidia/pytorch:24.03-py3
local_file=daily_container.tar
ssh_port=1228
volume_map="-v `pwd`:/workspace"

set -x
if [[ -f $local_file ]]; then
    docker load -i $local_file
else
    docker pull $image_name
fi
container_name="container-`date +%s`"
nvidia-docker run -it -p $ssh_port:22 --name $container_name    \
        $volume_map $image_name bash
echo "container is going to be saved in 30 seconds..."
sleep 30
if [[ $? == 0 ]]; then
    echo "container is being saved..."
    docker commit $container_name $image_name
    docker save -o $local_file $image_name
    docker rm $container_name
    echo "container has been saved..."
fi
