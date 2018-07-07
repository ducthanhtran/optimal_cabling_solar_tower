source /home/duc/virtualenvs/all/bin/activate

python ./src/main.py --input ./data/instances/PS10.csv \
                     --edge-cost-data 54 \
                     --trench-cost 50 \
                     --output-graph /home/duc/o.png \
                     --output-pkl /home/duc/o.pkl \
                     --output-init-graph /home/duc/o-init.png \
                     --output-init-pkl /home/duc/o-init.pkl
deactivate
