imgs=(*/)
for ((i=0; i<=${#imgs[@]}-1; i++)); do echo ${imgs[$i]}; grep sigma ${imgs[$i]}/OUTCAR | tail -1 | awk '{print $NF}'; done
