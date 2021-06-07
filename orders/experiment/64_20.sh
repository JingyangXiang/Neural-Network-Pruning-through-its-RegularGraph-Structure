for ((i=0;i<5;i++))
do
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_20 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/20/M0.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_20 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/20/M84.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_20 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/20/M95.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_20 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/20/M105.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_20 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/20/M129.npy
done
