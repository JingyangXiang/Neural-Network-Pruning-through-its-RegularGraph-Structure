for ((i=0;i<5;i++))
do
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_18 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/18/M0.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_18 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/18/M79.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_18 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/18/M97.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_18 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/18/M108.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_18 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/18/M119.npy
done
