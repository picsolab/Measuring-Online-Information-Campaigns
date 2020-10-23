import pickle
import numpy as np
import os
import sys

## create followers/friends data by merging csv files of followers/friends collected from Twitter.
def createFollowersFriendsData(campaign, connection_type):
    input_folder = 'data/{}/ea_{}_raw/'.format(campaign, connection_type)
    file_names = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            file_names.append(input_folder + filename)
            print(filename)
            sys.stdout.flush()
    
    user_connections = {}
    
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                lis = line.split(",")
                lis[-1] = lis[-1].strip()
                user_connections[lis[0].strip()] = lis[1:]
    
    out_file = '../data/social_media/{}/ea_{}_2020.pkl'.format(campaign, connection_type)
    pickle.dump(user_connections, open(out_file, 'wb'))

## Create the subset of followers/friends list for each user to have this info at the time when data collected.
def createPartialConnectionListData(campaign, connection_type, year):
    connection_list_file = '../data/social_media/{}/ea_{}_2020.pkl'.format(campaign, connection_type)
    out_file = '../data/social_media/{}/ea_{}_{}.pkl'.format(campaign, connection_type, str(year))
    connection_list = pickle.load(open(connection_list_file, 'rb'))
    users = pickle.load(open('../data/social_media/{}/all_users_by_relevant_videos.pkl'.format(campaign), 'rb'))
    connection_list_partial = {}
    print("num_users_with_old_counts:", len(connection_list.keys()))
    num_diffs = []
    perc_diffs = []
    for user_id in connection_list:
        past_num_connections = int(users[user_id]['_source']['followers_count']) if connection_type == 'followers' else int(users[user_id]['_source']['friends_count'])
        cur_num_connections = len(connection_list[user_id])
        if past_num_connections >= 0:
            if past_num_connections <= cur_num_connections:
                connection_list_partial[user_id] = connection_list[user_id][(cur_num_connections-past_num_connections):]
            else:
                connection_list_partial[user_id] = connection_list[user_id].copy()
                num_diffs.append(past_num_connections - cur_num_connections)
                perc_diffs.append((float(past_num_connections - cur_num_connections) / past_num_connections)*100)
        else:
            print(user_id, past_num_connections, cur_num_connections)
    
    print("num_users_with_new_counts:", len(connection_list_partial.keys()))
    
    num_diffs = np.array(num_diffs)
    perc_diffs = np.array(perc_diffs)
    print('# of users who lost followers:', len(num_diffs))
    print('mean (num):', np.mean(num_diffs))
    print('median (num):', np.median(num_diffs))
    print('mean (perc):', np.mean(perc_diffs))
    print('median (perc):', np.median(perc_diffs))
    
    pickle.dump(connection_list_partial, open(out_file, 'wb'))

## create followers/friends data by merging csv files of followers/friends collected from Twitter.
def mergeGraphWeightFiles(campaign, connection_type, year):
    input_folder = '../data/social_media/{}/graph_edges/{}/'.format(campaign, str(year))
    input_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if connection_type in filename:
                input_files.append(input_folder + filename)
                print(input_folder + filename)
    
    out_file = '../data/social_media/{}/graph_edges/ea_{}_{}.txt'.format(campaign, connection_type, str(year))
    with open(out_file, 'w') as outfile:
        for fname in input_files:
            with open(fname) as infile:
                outfile.write(infile.read())
            print(fname, 'processed!')
            sys.stdout.flush()

## Find raw early adopters (may contain the ones banned or protected).
def saveRawEarlyAdoptersUids(campaign, adoption_ratio):
    tweets = pickle.load(open("../data/social_media/{}/all_tweets_by_relevant_videos.pkl", 'rb'))
    filtered_video_ids = pickle.load(open("../data/social_media/{}/all_videos_by_relevant_videos.pkl", 'rb'))
    filtered_video_ids = list(filtered_video_ids.keys())

    video_tweets = {}
    for tid in tweets:
        uid = tweets[tid]['_source']['user_id_str']
        timestamp = int(tweets[tid]['_source']['timestamp_ms'])
        created_at = tweets[tid]['_source']['created_at']
        original_video_ids = tweets[tid]['_source']['original_vids'].split(';')
        retweeted_video_ids = tweets[tid]['_source']['retweeted_vids'].split(';')
        quoted_video_ids = tweets[tid]['_source']['quoted_vids'].split(';')
        video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
        if 'N' in video_ids:
            video_ids.remove('N')
        for vid in video_ids:
            if vid in filtered_videos:
                if vid not in video_tweets:
                    video_tweets[vid] = [(uid, tid, timestamp, created_at)]
                else:
                    video_tweets[vid].append((uid, tid, timestamp, created_at))

    # Find early adopters for each video.
    early_adopters_tweets = {}
    early_adopters_raw = []
    for vid in video_tweets:
        users = [item[0] for item in video_tweets[vid]]
        tweets = [item[1] for item in video_tweets[vid]]
        timestamps = np.array([int(item[2]) for item in video_tweets[vid]])
        created_ats = np.array([item[3] for item in video_tweets[vid]])

        timestamps_idx = np.argsort(timestamps)
        timestamps_sorted = list(itemgetter(*timestamps_idx)(timestamps))
        users_sorted = list(itemgetter(*timestamps_idx)(users))
        tweets_sorted = list(itemgetter(*timestamps_idx)(tweets))
        created_ats_sorted = list(itemgetter(*timestamps_idx)(created_ats))

        #print(users_sorted)
        #print(len(set(users_sorted)))

        users_sorted_unique = []
        for uid in users_sorted:
            if uid not in users_sorted_unique:
                users_sorted_unique.append(uid)

        early_adoption_thr_idx = math.ceil(len(users_sorted_unique)*adoption_ratio)
        users_sorted_unique = users_sorted_unique[:early_adoption_thr_idx]
        early_adopters_raw.extend(users_sorted_unique)

    pickle.dump(early_adopters_raw, open('../data/social_media/{}/ea_users_uids_raw.pkl', 'wb'))

if __name__ == '__main__' :
    ## [abo | gun | blm]
    campaign = 'blm'
    ## [followers | friends]
    connection_type = 'followers'
    ## [2018 | 2020]
    year = 2018
    
    #saveRawEarlyAdoptersUids(campaign, 0.2):
    #createFollowersFriendsData(campaign, connection_type)
    #createPartialConnectionListData(campaign, connection_type, year)
    #mergeGraphWeightFiles(campaign, connection_type, year)
    