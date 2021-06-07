for ((i=0;i<5;i++))
do
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_16 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/16/M0.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_16 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/16/M82.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_16 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/16/M90.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_16 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/16/M108.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_16 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/16/M119.npy
done
