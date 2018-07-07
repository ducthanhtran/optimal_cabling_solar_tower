# parameters
#  1. edge-cost (incl. trench cost)
#  2. trench cost
#  3. partitions

source ~/virtualenvs/all/bin/activate

python ./src/main.py --input ./data/instances/PS10.csv \
                     --edge-cost-data 54 \
                     --trench-cost 50 \
                     --output-graph ./solutions/after_ls_hamilton_${1}_${3}.png \
                     --output-pkl ./pkl/after_ls_hamilton_${1}_${3}.png.pkl \
                     --output-init-graph ./solutions/initial_hamilton_${1}_${3}.png \
                     --output-init-pkl ./pkl/initial_hamilton_${1}_${3}.png \
                     --partitions ${3}
deactivate
