
for name in {'Ped2',}
do
    rm exp/${name}*
    for th in {0.5,}
    do
        echo ' '
        echo '=============== start new loop ==============='
        python yolo_AD.py --threshold ${th} --dataset /data/VAD/${name}
    done
done
