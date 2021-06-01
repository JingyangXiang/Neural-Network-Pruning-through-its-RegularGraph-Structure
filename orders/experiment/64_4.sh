for ((j=0;i<5;i++))
do
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M0.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M31.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M58.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M83.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M108.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M133.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M160.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M186.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M206.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M231.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M253.npy
python main.py --config configs/experiments/$3-graph-$1.yaml --multigpu $2 --name 64_4 --first-layer-dense --last-layer-dense --set-connection --freeze-subnet --matrix Adjacency/64/4/M269.npy
done