
EPISODE=3000
for i in $(seq 1 0.5 10)
do 
    FILENAME="mm_$i.result"
    echo "" > $FILENAME    
    (time stdbuf -oL python3 -u reinforce.py --mm --omega $i --episode $EPISODE) &>> $FILENAME &
done

for i in $(seq 1 0.5 10)
do 
    FILENAME="softmax_$i.result"
    echo "" > $FILENAME    
    (time stdbuf -oL python3 -u reinforce.py --beta $i --episode $EPISODE) &>> $FILENAME &
done
echo "waiting..."
wait
echo "done"