import subprocess
import sys
import pickle
import json
import argparse
import os

def createChunks(seq, size, campaign):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    
    data_dir = "data/{}/".format(campaign)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    pickle.dump(newseq, open(os.path.join(data_dir, 'nonsuspended_uid_chunks.pkl'), 'wb'))
    return newseq

parser = argparse.ArgumentParser(description='')
parser.add_argument('--campaign', default='None', type=str)
args = parser.parse_args()
campaign = args.campaign

keys = None
with open('data/api_keys.json') as json_file:
    keys = json.load(json_file)

users = pickle.load(open('../data/social_media/{}/ea_users_suspended_or_not.pkl'.format(campaign), 'rb'))
non_suspended_user_ids = [key for (key, value) in users.items() if value == '']
user_id_chunks = createChunks(non_suspended_user_ids, len(keys), campaign)

#print("total # of non-suspended-users: ", len(non_suspended_user_ids))


