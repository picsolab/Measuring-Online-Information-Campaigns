import tweepy
import json
import re
import time
import sys
import pickle
import argparse

def getKey(chunk_no):
    keys = None
    with open('data/api_keys.json') as json_file:
        keys = json.load(json_file)
    return keys[chunk_no]

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)

def getFollowers(uid):#total=100
    backoff_counter = 1
    while True:
        try:
            user_followers = []
            for item in limit_handled(tweepy.Cursor(api.followers_ids, user_id =uid, count = 5000, cursor=-1, stringify_ids=True).items()):
                user_followers.append(item)
            return (user_followers)
            break
        except tweepy.TweepError as e:
            try:
                error_code = e.response.status_code
                return (error_code)
                break
            except:
                print(uid, " : ", e.reason, 'key:', key['APP_NAME'])
                sys.stdout.flush()
                time.sleep(backoff_counter * 60)
                backoff_counter += 1
                continue

def getFriends(uid):
    backoff_counter = 1
    while True:
        try:
            user_friends = []
            for item in limit_handled(tweepy.Cursor(api.friends_ids, user_id =uid, count = 5000, cursor=-1, stringify_ids=True).items()):
                user_friends.append(item)
            return (user_friends)
            break
        except tweepy.TweepError as e:
            try:
                error_code = e.response.status_code
                return (error_code)
                break
            except:
                print(uid, " : ", e.reason, 'key:', key['APP_NAME'])
                sys.stdout.flush()
                time.sleep(backoff_counter * 60)
                backoff_counter += 1
                continue
                

# Main function
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--campaign', default='None', type=str)
    parser.add_argument('--chunk_no', default='None', type=int)
    parser.add_argument('--type', default='None', type=str)
    args = parser.parse_args()
    campaign = args.campaign
    chunk_no = int(args.chunk_no)
    connection_type = args.type
    
    #read your twitter account list of userids
    non_suspended_user_ids = pickle.load(open('data/{}/nonsuspended_uid_chunks.pkl'.format(campaign), 'rb'))
    user_ids_chunk = non_suspended_user_ids[chunk_no]
    
    key = getKey(chunk_no)
    
    auth = tweepy.OAuthHandler(key['CONSUMER_KEY'], key['CONSUMER_SECRET'])
    auth.set_access_token(key['ACCESS_TOKEN'], key['ACCESS_TOKEN_SECRET'])

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    #api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=2, retry_delay=15)    
    
    #write out
    data_dir = "data/{}/{}_raw/".format(campaign, connection_type)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_name = os.path.join(data_dir, '{}.csv'.format(str(chunk_no)))
    outfile = open(file_name, "w")
    
    if connection_type == 'followers':
        count = 0
        for user_id in user_ids_chunk:
            follower_list = getFollowers(user_id)
            if type(follower_list) == list:
                print("count:", count, "user_id:", user_id, "follower_count:", len(follower_list), 'key:', key['APP_NAME'])
                sys.stdout.flush()
                result_lst_str = [user_id] + follower_list
                line_string = ",".join(result_lst_str)+"\n"
                outfile.write(line_string)
            else:
                print("error: ", user_id, follower_list, 'key:', key['APP_NAME'])
                sys.stdout.flush()
            count+=1
    
    elif connection_type == 'friends':
        count = 0
        for user_id in user_ids_chunk:
            friend_list = getFriends(user_id)
            if type(friend_list) == list:
                print("count:", count, "user_id:", user_id, "friend_count:", len(friend_list), 'key:', key['APP_NAME'])
                sys.stdout.flush()
                result_lst_str = [user_id] + friend_list
                line_string = ",".join(result_lst_str)+"\n"
                outfile.write(line_string)
            else:
                print("error:", user_id, friend_list, 'key:', key['APP_NAME'])
                sys.stdout.flush()
            count+=1
            
    
    outfile.close() 
    
