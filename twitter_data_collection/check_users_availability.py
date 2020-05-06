
tokens = [
            [
                "",
                "",
                "",
                ""
            ],
            [
                "",
                "",
                "",
                ""
            ],
            [
                "",
                "",
                "",
                ""
            ],
            [
                "",
                "",
                "",
                ""
            ]
        ]
   

import json
import os
import random
import requests
from requests_oauthlib import OAuth1
import glob
import time
import pickle
from pathlib import Path
import sys
import gzip
import glob

import requests.packages.urllib3

requests.packages.urllib3.disable_warnings()


class KeyChain(object):
    def __init__(self, tokens):
    #mengdi's key 1-5
        self.tokens = tokens
        self.currentKey = 0

    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def get_oauth(self):
        oauth = OAuth1(self.tokens[self.currentKey][0],
                       client_secret=self.tokens[self.currentKey][1],
                       resource_owner_key=self.tokens[self.currentKey][2],
                       resource_owner_secret=self.tokens[self.currentKey][3])
        return oauth

    def query(self, _oauth, _query):

        queryUrl = _query
        return requests.get(queryUrl, auth=_oauth)

    def __call__(self, query):
        responseJson = None
        while responseJson is None:
            oauth = self.get_oauth()
            buffer = -1
            while buffer < 0:
                response = self.query(oauth, query)

                try:
                    buffer = int(response.headers['x-rate-limit-remaining'])
                except Exception:
                    print('failed to get Rate Limit, retrying ...')
                    sys.stdout.flush()
                    return response

            if buffer % 5 == 0:
                #print('{} remains for token {}...'.format(buffer, self.currentKey))
                sys.stdout.flush()

            if buffer == 0:
                if self.currentKey < len(self.tokens) - 1:
                    print('Token Change from {} to {}'.format(self.currentKey, self.currentKey |+ 1))
                    sys.stdout.flush()
                    self.currentKey += 1
                    continue
                else:
                    current = round(time.time())
                    self.currentKey = 0
                    _oauth = self.get_oauth()
                    resetTime = -1
                    while resetTime < 0:
                        _response = self.query(_oauth, query)
                        try:
                            resetTime = int(_response.headers['x-rate-limit-reset'])
                        except Exception:
                            
                            print('failed to get reset time, retrying ...')
                            sys.stdout.flush()
                            continue

                    if current < resetTime + 200:
                        waitTime = resetTime - current + 200
                        print('wait for {} secs until quota reset...'.format(waitTime))
                        sys.stdout.flush()
                        time.sleep(waitTime)

                    continue
            else:
                responseJson = response.json()
        return responseJson

if __name__ == '__main__':
    keyC = KeyChain(tokens)
    #response = keyC(query='https://api.twitter.com/1.1/followers/ids.json?cursor={CUR}&count=5000&user_id={UID}'.format(CUR=-1, UID=uid))
    #response = keyC(query='https://api.twitter.com/1.1/users/lookup.json?user_id={UID}'.format(UID='814423198364147713,6253282,224499494'))
    #crawled_data = response.json()
    #print(response)
    
    campaign = 'gun'
    
    path_inp = 'data/{}/{}_all_users_additional_user_ids.pkl'.format(campaign, campaign)
    path_out = 'data/{}/{}_all_users_additional_users_suspended_or_not.pkl'.format(campaign, campaign)
    
    user_ids = pickle.load(open(path_inp, 'rb'))
    user_ids = list(user_ids.keys())
    print('#total users to be checked: {}'.format(len(user_ids)))
    sys.stdout.flush()

    cnt = 1
    data = {}
    chunks = keyC.chunks(user_ids, 90)
    for chunk in chunks:
        ## do query
        id_chunk = [str(uid) for uid in chunk]
        uids = ','.join(id_chunk)
        try:
            retrieved_user_objs = keyC(query='https://api.twitter.com/1.1/users/lookup.json?user_id={uids}'.format(uids=uids))
            if 'errors' not in retrieved_user_objs:
                for user_obj in retrieved_user_objs:
                    uid = user_obj['id_str']
                    protected = user_obj['protected']
                    if protected:
                        data[uid] = 'protected'
                    else:
                        data[uid] = ''
                    for uid in id_chunk:
                        if uid not in data:
                            data[uid] = 'suspended'
            else:
                print('--- All users in the chunk are suspended! ---')
                print(retrieved_user_objs)
                sys.stdout.flush()
                
        except Exception as e:
            print('Error:', e)
            sys.stdout.flush()
            pass
        
        if cnt % 10 == 0: 
            print("#users queried: {}".format(cnt*90))
            sys.stdout.flush()
        
        cnt += 1
    
    print("#users queried: {}".format(cnt*90))
    sys.stdout.flush()
    
    try:
        pickle.dump(data, open(path_out, 'wb'))
        print('dumped {}, data_len: {}'.format(path_out, len(data.keys())))
        sys.stdout.flush()
    except Exception as e:
        print('failed to dump {}, msg: {}'.format(path_out, str(e)))
        print(len(data.keys()))
        sys.stdout.flush()
    
