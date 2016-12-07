#!/bin/bash  

for i in $@; do
    while read line1  
    do  
        echo -e $line1 
    done < $i
done
