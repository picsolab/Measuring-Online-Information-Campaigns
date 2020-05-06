import pickle
import numpy as np
import os
import sys
import multiprocessing as mp
import time
import math

def createChunks(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

## create Shared Audience Measure Weights File based on the share followers/friends of users.
def createSAMWeights(campaign, start_idx, end_idx, keys, connection_type, connection_list, year):
    print(campaign.upper() + ": (" + str(start_idx) + "--" + str(end_idx) + ") started!")
    sys.stdout.flush()
    
    out_file_name = 'data/{}/graph_edges/sam/{}/{}_{}_{}.txt'.format(campaign, str(year), connection_type, str(start_idx), str(end_idx))
    out_file = open(out_file_name, "w")
    
    for i in range(start_idx, end_idx):
        for j in range(i+1, len(keys)):
            _intersect = len(list(set(connection_list[keys[i]]).intersection(set(connection_list[keys[j]]))))
            if(_intersect != 0):
                _union = (len(connection_list[keys[i]]) + len(connection_list[keys[j]])) - _intersect
                _jaccard = 0.
                if(_union != 0):
                    _jaccard = float(_intersect) / _union
                    message = keys[i] + " " + keys[j] + " " + ('%.10f' % _jaccard) + "\n"
                    #print(message)
                    out_file.write(str(message))
                    out_file.flush()
        
        if i % 10 == 0:
            print(campaign.upper() + ": (" + str(start_idx) + "--" + str(end_idx) + ") " + str(i) + " processed!")
            sys.stdout.flush()
    
    out_file.close()
    print(campaign.upper() + ": (" + str(start_idx) + "--" + str(end_idx) + ") finished!")
    sys.stdout.flush()


def main():
    ## [blm | gun | abo | cli | img]
    campaign = 'gun_ea_20'
    ## [followers | friends]
    connection_type = 'followers'
    ## [2017 | 2018 | 2020]
    year = 2018
    ## [sam | pmi]
    method_type = "sam"
    num_workers = 15
    
    followers_friends_list_file = 'data/{}/{}_{}.pkl'.format(campaign, connection_type, str(year))
    connection_list = pickle.load(open(followers_friends_list_file, 'rb'))
    print(followers_friends_list_file + " read!")
    sys.stdout.flush()
    
    users = {}
    for user in connection_list:
        for connection in connection_list[user]:
            users[connection] = 1
    len_connections = len(users.keys())
    print("# of unique " + connection_type + ":", len_connections)
    sys.stdout.flush()
    
    keys = list(connection_list.keys())
    chunks = createChunks(keys, num_workers)
    
    # Dispatch all the jobs
    jobs = []
    for k in range(num_workers):
        start_idx = keys.index(chunks[k][0])
        end_idx = None
        if k == (num_workers - 1):
            end_idx = len(keys)
        else:
            end_idx = keys.index(chunks[k+1][0])
        
        job = mp.Process(target = createSAMWeights, args=(campaign, start_idx, end_idx, keys, connection_type, connection_list, year))
        jobs.append(job)
        job.start()
        #jobs.append(job)
    # Make sure the jobs are finished
    for job in jobs:
        job.join()
    

if __name__ == '__main__' :
    main()