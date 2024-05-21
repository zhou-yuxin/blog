image_name=nvcr.io/nvidia/pytorch:24.03-py3
local_file=daily_container.tar
volume_map="-v `pwd`:/workspace"

ret=`docker images -q $image_name`
if [[ -n $ret ]]; then
    :
elif [[ -f $local_file ]]; then
    docker load -i $local_file
else
    docker pull $image_name
fi
container_name="C-`date +%s`"
nvidia-docker run -it --name $container_name    \
        $volume_map $image_name bash
read -p "commit and save (y/n)?" save
if [[ $save == y ]]; then
    echo "container is being saved..."
    docker commit $container_name $image_name
    docker save -o $local_file $image_name
    docker rm $container_name
    echo "container has been saved..."
fi
