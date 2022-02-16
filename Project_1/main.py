from UMedLS import UMedLS
import json
import os
import time

def main():

    # initialize an instance of the class
    umls_ = UMedLS()

    # find cycles
    starting_nodes = ['C2937260', 'C0003850', 'C2937260', 'C0005586', 'C0029456']
    paths_ = {}
    for starting_node in starting_nodes:
        t = time.time()
        paths_[starting_node] = {'cycle': umls_.dfs(starting_node, starting_node, 0),
                                 'search time': time.time() - t}
        print('Cycle from %s:'%(starting_node))
        print(paths_[starting_node]['cycle'])
        print('Time required to find cycle:')
        print('%f secs'%(paths_[starting_node]['search time']))
        print('\n')

    # write text file
    with open('OUTPUT.txt', 'w') as f:
        for cui in paths_:
            f.write(paths_[cui]['cycle'])
            f.write('\n')

if __name__ == "__main__":
    main()