import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="UGRec experiment.")
    parser.add_argument('--dataset', nargs='?', default='game',
                        help='Choose a dataset from {xxx, xxx, xxx}')
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--Ks', nargs='?', type=int, default=20,
                        help='Top k(s) recommend')
    parser.add_argument('--margins', nargs='?', default='[1.5, 1.5, 1.5]',
                        help='Margins for major interactive space, directed side information spaces and undirected side information spaces.')

    return parser.parse_args()
