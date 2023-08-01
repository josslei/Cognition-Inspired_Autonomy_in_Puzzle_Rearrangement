import sys

import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 visualize_loss.py <path_to_losses>')
        exit()
    
    path_losses = sys.argv[1]

    with open(path_losses, 'r') as fp:
        losses = fp.readlines()
    losses = [float(l) for l in losses]
    epochs = [i for i, _ in enumerate(losses)]

    plt.plot(epochs, losses)
    plt.savefig('./loss_graph.png')
