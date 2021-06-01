for ((j=0;i<5;i++))
do
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M0.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M106.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M141.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M163.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M187.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_8 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/8/M209.npy
done