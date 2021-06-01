for ((j=0;i<5;i++))
do
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_12 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/12/M0.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_12 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/12/M82.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_12 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/12/M127.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_12 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/12/M111.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_12 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/12/M149.npy
done