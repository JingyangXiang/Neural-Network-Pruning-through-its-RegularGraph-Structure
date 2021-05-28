from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch
import networkx as nx

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-paths', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')

    args = parser.parse_args()
    paths = [os.path.join(args.in_paths, path) for path in os.listdir(args.in_paths)]
    index = 0
    for path in paths:
        if 'logs' in path and os.path.exists(path):
            path = path
        elif os.path.exists(os.path.join(path, 'logs')):
            path = os.path.join(path, 'logs')
        else:
            continue
        dirs = os.listdir(path)
        if len(dirs)==1:
            in_path = os.path.join(path,dirs[0])
        else:
            for dir in dirs:
                if 'events' in dir:
                    in_path = os.path.join(path, dir)
                    break
        out_path = os.path.join(path,'logs.csv')
        event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        keys = event_data.scalars.Keys()  # get all tags,save in a list
        # print(keys)
        df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
        for key in tqdm(keys):
            # print(key)
            if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
                df[key] = pd.DataFrame(event_data.Scalars(key)).value

        df.to_csv(out_path)

        print(f"{index} Tensorboard data {out_path} exported successfully")
        index = index + 1


        model_path = path.replace('logs', 'checkpoints')
        for dir in os.listdir(model_path):
            if 'model_best.pth' in dir:
                model_paths = os.path.join(model_path, 'model_best.pth')
                model = torch.load(model_paths,map_location=torch.device('cpu'))
                for key in model['state_dict'].keys():
                    if 'score' in key:
                        matrix = model['state_dict'][key].cpu().numpy().squeeze()
                        assert len(matrix.shape)==2
                        matrix = matrix[::matrix.shape[0]//64, ::matrix.shape[1]//64]
                        assert matrix.shape[0]==64 and matrix.shape[1]==64
                        graph = nx.from_numpy_matrix(matrix)
                        lengthpath = nx.average_shortest_path_length(graph)
                        if not os.path.exists(os.path.join(path, f'{lengthpath}')):
                            os.makedirs(os.path.join(path, f'{lengthpath}'))
                        print(f'length of path {lengthpath}')
                        break

if __name__ == '__main__':
    main()