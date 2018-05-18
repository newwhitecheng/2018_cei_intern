rm single_*log
for i in `seq $1`
do
    python client_single.py $i $1 -w 10.30.5.243 --port 8000 --user user_330 --pass pass@#2249 1 >single_$i.log  2>&1 &
done
#python client_single.py 2 3 -w 10.30.5.243 --port 8000 --user user_330 --pass pass@#2249 1 >single_2.log  2>&1 &
#python client_single.py 3 3 -w 10.30.5.243 --port 8000 --user user_330 --pass pass@#2249 1 >single_3.log  2>&1 &

