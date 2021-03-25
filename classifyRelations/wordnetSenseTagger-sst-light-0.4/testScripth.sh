#!/bin/bash

n=0
maxjobs=14
for i in $(pwd)/*.txt ; do
    # ( DO SOMETHING ) &
    echo $i &
#echo $i.tags &
    # limit jobs

    echo $((++n))
    if (( $(($((++n)) % $maxjobs)) == 0 )) ; then
        wait # wait until all have finished (not optimal, but most times good enough)
        echo $n wait
    fi
done

