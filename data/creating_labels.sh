cat train.txt dev.txt test.txt | cut -f 2 | cut -d "_" -f 1 | grep -v "^$"| sort | uniq > labels.txt
