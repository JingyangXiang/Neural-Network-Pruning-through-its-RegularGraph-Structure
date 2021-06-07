for ((i=0;i<5;i++))
do
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M0.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M57.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M76.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M95.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M107.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_14 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/14/M119.npy
done
