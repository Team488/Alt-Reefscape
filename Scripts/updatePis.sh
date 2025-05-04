targets=(
    photonvisionfrontleft
    photonvisionfrontright
    photonvisionrearleft
    photonvisionrearright
)
for target in "${targets[@]}"
do
sshpass -p 'raspberry' rsync -avz / pi@${target}:/file/location/for/{xbot}
done
