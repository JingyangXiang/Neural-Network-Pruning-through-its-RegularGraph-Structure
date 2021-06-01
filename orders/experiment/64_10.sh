for ((j=0;i<5;i++))
do
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_10 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/10/M0.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_10 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/10/M110.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_10 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/10/M137.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_10 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/10/M157.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_10 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/10/M179.npy
done