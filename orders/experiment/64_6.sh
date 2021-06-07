for ((i=0;i<5;i++))
do
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M0.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M66.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M115.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M138.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M163.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M188.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M215.npy
    python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_6 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/6/M239.npy
done
