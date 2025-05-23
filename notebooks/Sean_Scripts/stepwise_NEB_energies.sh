imgs=(*/)
vgrep () { grep "$1" $(find . -maxdepth 1 -name "vasp_out*" | xargs ls -t | head -1); } # assumes your vasp output files are named vasp_out...
steps=$(vgrep F= | tail -1 | awk '{print $1}')
filename='NEB_stepwise_energies.dat' # can rename if ya want
touch $filename
for step in $(seq 1 $steps)
do
echo $step >> $filename
for ((i=0; i<=${#imgs[@]}-1; i++)); do grep 'entropy=' ${imgs[$i]}/OUTCAR | head -${step} | tail -1 | awk '{print $NF}' >> $filename; done
done
