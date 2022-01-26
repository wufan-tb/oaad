
for name in {'SHTech',}
do
    rm exp/${name}*
    for th in {0.8,}
    do
        echo ' '
        echo '=============== start new loop ==============='
        python yolo_slowfast_AD.py --threshold ${th} --dataset /data/VAD/${name}
    done
done
