#!/bin/bash

n=0
maxjobs=14
for i in $(pwd)/rawSentences.txt ; do #/user/socherr/scr/projects/asap/data/essasyTxt/*.txt ; do
    # ( DO SOMETHING ) &
#./sst multitag $i 0 0 DATA/GAZ/gazlistall_minussemcor ./MODELS/WSJPOSc_base_20 DATA/WSJPOSc.TAGSET ./MODELS/SEM07_base_12 ./DATA/WNSS_07.TAGSET ./MODELS/WSJc_base_20 ./DATA/WSJ.TAGSET > $i.tags &
./sst multitag-line $i 0 0 DATA/GAZ/gazlistall_minussemcor ./MODELS/WSJPOSc_base_20 DATA/WSJPOSc.TAGSET ./MODELS/SEM07_base_12 ./DATA/WNSS_07.TAGSET ./MODELS/WSJc_base_20 ./DATA/WSJ.TAGSET > $i.tags &
#echo $i.tags &
    # limit jobs
    if (( $(($((++n)) % $maxjobs)) == 0 )) ; then
        wait # wait until all have finished (not optimal, but most times good enough)
        echo $n wait
    fi
done

