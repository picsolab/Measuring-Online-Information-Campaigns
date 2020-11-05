import json
import pickle
import numpy as np
import pandas as pd
import math
import random
random.seed(777)
import csv
from datetime import datetime, timezone
import copy
import dateutil.parser as dt
from operator import itemgetter
import os
import location
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import OrderedDict 
from operator import getitem
import networkx as nx
import math
import graph_ops
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, kruskal, ks_2samp
from empath import Empath
from src.language.text_parser import EkphrasisParser
from src.language import liwc

thresholds = {
    "abo": {"left": 0.53784, "right": 0.74940},
    "gun": {"left": 0.47838, "right": 0.71606}, 
    "blm": {"left": 0.52373, "right": 0.69474}
}

community_resolution_param = {
    "abo": 1,
    "gun": 0.8, 
    "blm": 1
}

class Tweet:
    def __init__(self, util):
        
        self.util = util
        
        ## Initial tweets and videos retrieved by topical keywords
        self.initial_tweets_path = 'data/social_media/{}/tweets_by_keywords.pkl'.format(self.util.campaign)
        self.initial_videos_path = 'data/social_media/{}/videos_by_keywords.pkl'.format(self.util.campaign)
        
        ## All relevant tweets, users, videos by relevant videos
        self.tweets_path = 'data/social_media/{}/all_tweets_by_relevant_videos.pkl'.format(self.util.campaign)
        self.users_path = 'data/social_media/{}/all_users_by_relevant_videos.pkl'.format(self.util.campaign)
        self.videos_path = 'data/social_media/{}/all_videos_by_relevant_videos.pkl'.format(self.util.campaign)
        
        ## Early adopters' status about available, banned or protected. (Early adopters are not clean yet.)
        self.ea_users_suspended_path = 'data/social_media/{}/ea_users_suspended_or_not.pkl'.format(self.util.campaign)
       
        ## Clean (i.e. available) early adopters' tweets, user_ids, followers, locations, leaning labels and inferred scores.
        self.ea_tweets_path = 'data/social_media/{}/ea_tweets.pkl'.format(self.util.campaign)
        self.ea_users_ids_path = 'data/social_media/{}/ea_users_ids.pkl'.format(self.util.campaign)
        self.ea_users_followers_path = 'data/social_media/{}/ea_followers_{}.pkl'.format(self.util.campaign, str(self.util.year))
        self.ea_users_locs_path = 'data/social_media/{}/ea_users_locs.pkl'.format(self.util.campaign)
        self.ea_users_sub_followings_path = 'data/social_media/{}/ea_users_sub_followings_{}.pkl'.format(self.util.campaign,
                                                                                                   str(self.util.year))
        
        self.left_seed_political_hashtags_path = 'data/leaning_keywords/left_political.txt'
        self.right_seed_political_hashtags_path = 'data/leaning_keywords/right_political.txt'
        self.left_extended_political_hashtags_path = 'data/social_media/{}/left_political_hashtags_extended.txt'.format(self.util.campaign)
        self.right_extended_political_hashtags_path = 'data/social_media/{}/right_political_hashtags_extended.txt'.format(self.util.campaign)
        self.ea_seed_users_leanings_labels_path = 'data/social_media/{}/ea_seed_users_leanings_labels.pkl'.format(self.util.campaign)
        self.ea_users_inferred_leanings_scores_path = 'data/social_media/{}/ea_users_inferred_leanings_scores.pkl'.format(self.util.campaign)
        
        self.users_locs_path = 'data/social_media/{}/all_users_locs.pkl'.format(self.util.campaign)
        
        ## Early adopters' communities path
        self.ea_communities_path = 'data/social_media/{}/ea_communities_{}_res_{}.pkl'.format(self.util.campaign,
                                                                                              str(self.util.year),
                                                                                              community_resolution_param[self.util.campaign])
        
        ## Video leaning thresholds
        self.predefined_video_leaning_thr = thresholds[self.util.campaign]
        
        ## Video leaning thresholds
        self.ea_network_structural_measures_path = "data/social_media/{}/ea_network_structural_measures.pkl".format(self.util.campaign)
        self.ea_engagement_measures_path = "data/social_media/{}/ea_engagement_measures.pkl".format(self.util.campaign)
        self.ea_temporal_measures_path = "data/social_media/{}/ea_temporal_measures.pkl".format(self.util.campaign)
        self.ea_language_liwc_measures_path = "data/social_media/{}/ea_language_liwc_measures.pkl".format(self.util.campaign)
        self.ea_language_empath_measures_path = "data/social_media/{}/ea_language_empath_measures.pkl".format(self.util.campaign)
        self.ea_cascade_measures_path = 'data/social_media/{}/ea_cascade_measures.pkl'.format(self.util.campaign)
        #self.ea_geo_measures_path = "data/social_media/{}/ea_geo_measures.pkl".format(self.util.campaign)
        
        self.bin_size = self.util.bin_size
        self.daily_dates = self.util.daily_dates
        self.aggregated_dates = self.util.getAggregatedDates()
        self.active_videos_ids = 'data/from_anu/_active_videos.pkl'
        self.video_annotations_path = 'data/from_anu/v1/{}_video_annotations.csv'.format(self.util.campaign)
        
    
    def getTweetVolumeDistribution(self, data):

        daily_counts = np.zeros((len(self.daily_dates)))
        for item in data.values():
            created_at = item['_source']['created_at']
            daily_counts[self.daily_dates.index(created_at)] += 1
        
        print('# items:', len(data.keys()))
        #print('# leap_items:', daily_counts[-1])
        
        ## weekly aggregation
        aggregated_counts = self.util.getAggregatedCounts(daily_counts)
        print('# of total tweets:', np.sum(aggregated_counts))
        
        return aggregated_counts
    
    def getTweets(self):
        tweets = pickle.load(self.tweets_path, 'rb')
        return tweets
    
    ## Sort users based on the tweet volume they share
    def sortUsersByTweetVolume(self, tweets):
        users = {}
        for tweet in tweets:
            userId = tweet['_source']['userId']
            if userId in users:
                users[userId] += 1
            else:
                users[userId] = 1
        
        sorted_users = sorted(users.items(), key=itemgetter(1))
        
        for user in sorted_users:
            print(user)
    
    ## Sort users based on the tweet volume they share for a specific week
    def sortUsersByTweetVolumeForSpecificInterval(self, tweets, week_no):
        users = {}
        dates = self.aggregated_dates[week_no - 1].split(" ")
        start_date = np.datetime64(dates[0])
        end_date = start_date + np.timedelta64(self.bin_size, 'D')
        for tweet_id in tweets:
            created_at = np.datetime64(tweets[tweet_id]['_source']['created_at'])
            if created_at >= start_date and created_at < end_date:
                user_id = tweets[tweet_id]['_source']['userId']
                if user_id in users:
                    users[user_id] += 1
                else:
                    users[user_id] = 1
                    
        sorted_users = sorted(users.items(), key=itemgetter(1), reverse=True)
        
        for user in sorted_users:
            print(user)
    
    ## Sort videos based on the tweet volume they refer to
    def sortVideosByTweetVolume(self, tweets):
        videos = {}
        for tweet_id in tweets:
            video_ids = tweets[tweet_id]['_source']['videoId'].split(';')
            for video_id in video_ids:
                if video_id in videos:
                    videos[video_id] += 1
                else:
                    videos[video_id] = 1
        
        sorted_videos = sorted(videos.items(), key=itemgetter(1), reverse=True)
        
        for video_id in sorted_videos:
            print(video_id)
    
    ## Sort videos based on the tweet volume they refer to for a specific week
    def sortVideosByTweetVolumeForSpecificInterval(self, tweets, start_date, end_date):
        videos = {}
        for tweet_id in tweets:
            created_at = tweets[tweet_id]['_source']['created_at']
            if created_at >= start_date and created_at < end_date:
                original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
                retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
                quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
                video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
                if 'N' in video_ids:
                    video_ids.remove('N')
                for video_id in video_ids:
                    if video_id in videos:
                        videos[video_id] += 1
                    else:
                        videos[video_id] = 1
        
        sorted_videos = sorted(videos.items(), key=itemgetter(1), reverse=True)
        
        for video_id in sorted_videos:
            print(video_id)
        
        return sorted_videos
    
    ## Print tweets referring the the specific video within a specific week
    def printTweetsByVideoId(self, tweets, start_date, end_date, video_id):
        #user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            #if user_id in user_comm_pairs['assigned_com_memberships']:
            created_at = tweets[tweet_id]['_source']['created_at']
            if created_at >= start_date and created_at < end_date:
                original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
                retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
                quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
                video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
                if video_id in video_ids:
                    print(user_id)
                    if tweets[tweet_id]['_source']['original_hashtags'].strip() != 'N':
                        print('original_hashtags: ', tweets[tweet_id]['_source']['original_hashtags'])
                    if tweets[tweet_id]['_source']['original_text'].strip() != 'N':
                        print('original_text: ', tweets[tweet_id]['_source']['original_text'])
                    if tweets[tweet_id]['_source']['retweeted_hashtags'].strip() != 'N':
                        print('retweeted_hashtags: ', tweets[tweet_id]['_source']['retweeted_hashtags'])
                    if tweets[tweet_id]['_source']['retweeted_text'].strip() != 'N':
                        print('retweeted_text: ', tweets[tweet_id]['_source']['retweeted_text'])
                    if tweets[tweet_id]['_source']['quoted_hashtags'].strip() != 'N':
                        print('quoted_hashtags: ', tweets[tweet_id]['_source']['quoted_hashtags'])
                    if tweets[tweet_id]['_source']['quoted_text'].strip() != 'N':
                        print('quoted_text: ', tweets[tweet_id]['_source']['quoted_text'])
                    print('===========================================================================')
    
    
    def getUserTweetCount(self, tweets):
        users = {}
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            if user_id in users:
                users[user_id] += 1
            else:
                users[user_id] = 1
        
        print('num_users: ', len(users.keys()))    
        sorted_users = sorted(users.items(), key=itemgetter(1), reverse=False)
        return sorted_users
    
    ## Print random tweets of the users who share the tweets most
    def analyzeTweetsOfMostTweetedUsers(self, tweets):
        top_k_user = 20
        num_tweets_per_user = 5
        users_tweet_counts = self.getUserTweetCount(tweets)[-top_k_user:]
        userids = [pair[0] for pair in users_tweet_counts]
        users = {}
        for user_id in userids:
            users[user_id] = 0
        
        tweet_ids = list(tweets.keys())
        random.shuffle(tweet_ids)
        
        with open(self.util.campaign + '_random_tweets.csv' , 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            writer.writerow(["tweetid", "userid", "original_tweet", "original_hashtags", "retweeted_tweet", "retweeted_hashtags", "quoted_tweet", "quoted_hashtags"])
        
            for tweet_id in tweet_ids:
                user_id = tweets[tweet_id]['_source']['user_id_str']
                if user_id in users and users[user_id] < num_tweets_per_user:
                    print('user_id: ', user_id)
                    original_tweet = tweets[tweet_id]['_source']['original_text']
                    original_hashtags = tweets[tweet_id]['_source']['original_hashtags']
                    retweeted_tweet = tweets[tweet_id]['_source']['retweeted_text']
                    retweeted_hashtags = tweets[tweet_id]['_source']['retweeted_hashtags']
                    quoted_tweet = tweets[tweet_id]['_source']['quoted_text']
                    quoted_hashtags = tweets[tweet_id]['_source']['quoted_hashtags']
                    print('orginal_tweet: ', original_tweet)
                    if(original_hashtags.strip() != 'N'):
                        print('original_hashtags: ', original_hashtags)
                    if(retweeted_tweet.strip() != 'N'):
                        print('retweeted_tweet: ', retweeted_tweet)
                    if(retweeted_hashtags.strip() != 'N'):
                        print('retweeted_hashtags: ', retweeted_hashtags)
                    if(quoted_tweet.strip() != 'N'):
                        print('quoted_tweet: ', quoted_tweet)
                    if(quoted_hashtags.strip() != 'N'):
                        print('quoted_hashtags: ', quoted_hashtags)
                    
                    writer.writerow([tweet_id, user_id, original_tweet, retweeted_hashtags, retweeted_tweet, retweeted_hashtags, quoted_tweet, quoted_hashtags])
                    users[user_id] += 1
    
    
    ## Return available tweets, create available tweet subset from all explicit tweets if it does not exist.
    def getAvailableTweets(self, ea_ratio):
        if os.path.isfile(self.ea_tweets_path):
            available_ea_tweets = pickle.load(open(self.ea_tweets_path, 'rb'))
            return available_ea_tweets
        else:
            tweets = pickle.load(open(self.tweets_path, 'rb'))
            followers = pickle.load(open(self.ea_users_followers_path, 'rb'))
            filtered_video_ids = pickle.load(open(self.videos_path, 'rb'))
            filtered_video_ids = list(filtered_video_ids.keys())
            
            video_early_adopters = self.findEarlyAdopters(ea_ratio, tweets, filtered_video_ids)
            all_ea_tweet_ids = []
            cnt = 0
            for video_id in video_early_adopters:
                all_ea_tweet_ids.extend([item[1] for item in video_early_adopters[video_id]])
                
                cnt += len([item[1] for item in video_early_adopters[video_id]])
                #print(len([item[1] for item in video_early_adopters[video_id]]))
            
            #print('cnt:', cnt)
            #print('all_ea_tweet_ids:', len(all_ea_tweet_ids))
            #print('all_ea_tweet_ids:', len(set(all_ea_tweet_ids)))
            
            available_ea_tweets = {}
            for tid in all_ea_tweet_ids:
                uid = tweets[tid]['_source']['user_id_str']
                if uid in followers:
                    available_ea_tweets[tid] = copy.deepcopy(tweets[tid])
            
            available_ea_user_ids = {}
            for tid in available_ea_tweets:
                uid = available_ea_tweets[tid]['_source']['user_id_str']
                available_ea_user_ids[uid] = 0
        
            pickle.dump(available_ea_tweets, open(self.ea_tweets_path, 'wb'))
            pickle.dump(available_ea_user_ids, open(self.ea_users_ids_path, 'wb'))
            return available_ea_tweets
    
    
    ## Summarize users and tweets by suspended, protected and available.
    def summarizeUsersTweetsInfo(self, tweets):
        followers = pickle.load(open(self.ea_users_followers_path, 'rb'))
        
        num_avaiable_users = len(list(followers.keys()))
        print('# total available tweets:', len(tweets.keys()))
        
        num_original_tweets_from_available = 0
        num_retweets_from_available = 0
        num_quoted_from_available = 0
        num_replies_from_available = 0
        
        cnt=0
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            tweet_id_str = tweets[tweet_id]['_source']['tweet_id_str']
            retweeted_tweet_id_str = tweets[tweet_id]['_source']['retweeted_tweet_id_str']
            quoted_tweet_id_str = tweets[tweet_id]['_source']['quoted_tweet_id_str']
            reply_user_id_str = tweets[tweet_id]['_source']['reply_user_id_str']
            
            if retweeted_tweet_id_str == 'N' and quoted_tweet_id_str == 'N' and user_id in followers:
                num_original_tweets_from_available += 1
            elif retweeted_tweet_id_str != 'N' and user_id in followers:
                num_retweets_from_available += 1
            elif retweeted_tweet_id_str == 'N' and quoted_tweet_id_str != 'N' and user_id in followers:
                num_quoted_from_available += 1
            else:
                cnt+=1
            
            if reply_user_id_str != 'N':# and user_id in followers:
                num_replies_from_available += 1
            
        print('# available users:', num_avaiable_users, '# avaiable original tweets:', num_original_tweets_from_available, '# available retweets:', num_retweets_from_available, '# available quoted tweets:', num_quoted_from_available, '# available replies:', num_replies_from_available, 'cnt:', cnt)
            
    
    ## Summarize number of users w.r.t their number of followers.
    def summarizeFollowers(self):
        followers = pickle.load(open(self.ea_users_followers_path, 'rb'))
        
        follower_counts = np.array([len(item) for item in followers.values()])
        users = {}
        for user in followers:
            for follower in followers[user]:
                users[follower] = 1
        
        print('total # of followers:', np.sum(follower_counts), 'total # of unique followers:', len(users.keys()))
        
        a = follower_counts[follower_counts<500].shape[0]
        b = follower_counts[(follower_counts>=500) & (follower_counts<1000)].shape[0]
        c = follower_counts[(follower_counts>=1000) & (follower_counts<5000)].shape[0]
        d = follower_counts[(follower_counts>=5000) & (follower_counts<10000)].shape[0]
        e = follower_counts[follower_counts>=10000].shape[0]
        print(a, b, c, d, e)
        
        plt.hist(follower_counts, normed=False, bins=50)
        plt.ylabel('Follower counts');
        plt.show()
        
    
    ## Find early adaptors for videos.
    def findEarlyAdopters(self, adoption_ratio, tweets, filtered_videos):
        # Get all users, tweet for each video.
        # video_tweets[video_id] = [(user_id, tweet_id, timestamp, created_at), ...]
        video_tweets = {}
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            timestamp = int(tweets[tweet_id]['_source']['timestamp_ms'])
            created_at = tweets[tweet_id]['_source']['created_at']
            original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            for video_id in video_ids:
                if video_id in filtered_videos:
                    if video_id not in video_tweets:
                        video_tweets[video_id] = [(user_id, tweet_id, timestamp, created_at)]
                    else:
                        video_tweets[video_id].append((user_id, tweet_id, timestamp, created_at))
                        
        # Find early adopters for each video.
        # early_adopters_tweets[video_id] = [(user_id, tweet_id, timestamp, created_at), ...]
        early_adopters_tweets = {}
        for video_id in video_tweets:
            users = [item[0] for item in video_tweets[video_id]]
            tweets = [item[1] for item in video_tweets[video_id]]
            timestamps = np.array([int(item[2]) for item in video_tweets[video_id]])
            created_ats = np.array([item[3] for item in video_tweets[video_id]])
            #print(created_ats)
            timestamps_idx = np.argsort(timestamps)
            timestamps_sorted = list(itemgetter(*timestamps_idx)(timestamps))
            users_sorted = list(itemgetter(*timestamps_idx)(users))
            tweets_sorted = list(itemgetter(*timestamps_idx)(tweets))
            created_ats_sorted = list(itemgetter(*timestamps_idx)(created_ats))
            
            #print(users_sorted)
            #print(len(set(users_sorted)))
            
            users_sorted_unique = []
            for user_id in users_sorted:
                if user_id not in users_sorted_unique:
                    users_sorted_unique.append(user_id)
            
            early_adoption_thr_idx = math.ceil(len(users_sorted_unique)*adoption_ratio)
            users_sorted_unique = users_sorted_unique[:early_adoption_thr_idx]
            
            #print(users_sorted_unique)
            #print(len(users_sorted_unique))
            
            early_adopters_tweets[video_id] = []
            for i in range(len(users_sorted)):
                if users_sorted[i] in users_sorted_unique:
                    item = (users_sorted[i], tweets_sorted[i], timestamps_sorted[i], created_ats_sorted[i])
                    early_adopters_tweets[video_id].append(item)
                else:
                    break
        
        return early_adopters_tweets
    
    
    
    ###########################################################################
    ## USER Profile Enhancement operations ####################################
    ###########################################################################
    ## Find and assign location of the users from tweets and profiles.
    def assignUserLocations(self, tweets):
        users = pickle.load(open(self.users_path, 'rb'))
        available_users = list(set([tweets[tid]['_source']['user_id_str'] for tid in tweets]))
        users_locs = {}
        
        locations_from_tweets = {}
        locations_from_user_profiles = {}
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            
            user_loc = users[user_id]['_source']['location'].strip()
            est_user_loc = location.findLocation(user_loc)
            if user_id not in locations_from_user_profiles:
                locations_from_user_profiles[user_id] = est_user_loc
            
            print(user_loc, '-->', est_user_loc)
            
            tweet_cc = tweets[tweet_id]['_source']['original_countrycode'].strip()
            tweet_geoname = tweets[tweet_id]['_source']['original_geoname'].strip()
            tweet_loc = location.findLocationFromTweet(tweet_geoname, tweet_cc, est_user_loc, user_loc)
            if tweet_loc != None:
                if user_id in locations_from_tweets:
                    if tweet_loc in locations_from_tweets[user_id]:
                        locations_from_tweets[user_id][tweet_loc] += 1
                    else:
                        locations_from_tweets[user_id][tweet_loc] = 1
                else:
                    locations_from_tweets[user_id] = {}
                    locations_from_tweets[user_id][tweet_loc] = 1
        
        
        print(locations_from_tweets)
        
        for user_id in locations_from_tweets:
            locs = locations_from_tweets[user_id]
            sorted_locs = sorted(locs.items(), key=itemgetter(1), reverse=True)
            locations_from_tweets[user_id] = sorted_locs[0][0]
            
        print('-------------------------------------------------------------------')
        print(locations_from_tweets)
        
        #for user_id in users:
        for user_id in available_users:
            est_loc = None
            est_user_loc = locations_from_user_profiles[user_id]
            est_tweet_loc = locations_from_tweets.get(user_id)
            
            if est_user_loc != None and est_tweet_loc == None:
                est_loc = est_user_loc
            elif est_user_loc == None and est_tweet_loc != None:
                est_loc = est_tweet_loc
            elif est_user_loc != None and est_tweet_loc != None:
                if est_user_loc == 'United States' and est_tweet_loc in location.getStates():
                    est_loc = est_tweet_loc
                else:
                    est_loc = est_user_loc
            
            users_locs[user_id] = est_loc
        
        pickle.dump(users_locs, open(self.ea_users_locs_path, 'wb'))
    
    ## Find all hashtags in user profiles
    def getAllHashtags(self):
        users = pickle.load(open(self.users_path, 'rb'))
        available_users_ids = pickle.load(open(self.ea_users_ids_path, 'rb'))
        hashtags = {}
        for user_id in users:
            if user_id in available_users_ids:
                user_desc = users[user_id]['_source']['description']
                user_hashtags = list(set(re.findall(r"#(\w+)", user_desc)))
                user_hashtags = [tag.lower() for tag in user_hashtags]
                for tag in user_hashtags:
                    if tag not in hashtags:
                        hashtags[tag] = 1
                    else:
                        hashtags[tag] += 1
         
        sorted_hashtags = sorted(hashtags.items(), key=itemgetter(1), reverse=True)
        return sorted_hashtags
    
    ## Find the hastag co-occurence matrix and entropy w.r.t the left political and right political hashtags.
    def findHashtagCooccurrences(self):
        available_users_ids = pickle.load(open(self.ea_users_ids_path, 'rb'))
        users = pickle.load(open(self.users_path, 'rb'))
        
        all_hashtags = self.getAllHashtags()
        hashtags = [ht[0] for ht in all_hashtags if ht[1] > 3]
        
        left_political_hts = []
        with open(self.left_seed_political_hashtags_path, 'r') as f:
            inp = f.readlines()
            left_political_hts = [kw.rstrip() for kw in inp]
        
        for ht in left_political_hts:
            if ht in hashtags:
                hashtags.remove(ht)
        
        right_political_hts = []
        with open(self.right_seed_political_hashtags_path, 'r') as f:
            inp = f.readlines()
            right_political_hts = [ht.rstrip() for ht in inp]
        
        for ht in right_political_hts:
            if ht in hashtags:
                hashtags.remove(ht)
        
        cooccur_mat = np.zeros(shape=(2, len(hashtags)))
        ht_occur_vec = np.zeros(shape=(len(hashtags),))
        for user_id in users:
            if user_id in available_users_ids:
                user_desc = users[user_id]['_source']['description']
                user_hashtags = list(set(re.findall(r"#(\w+)", user_desc)))
                user_hashtags = [tag.lower() for tag in user_hashtags]
                left_hts_in_profile = []
                right_hts_in_profile = []
                for ht in user_hashtags:
                    if ht in left_political_hts:
                        left_hts_in_profile.append(ht)
                    elif ht in right_political_hts:
                        right_hts_in_profile.append(ht)
                
                for ht in user_hashtags:
                    if ht in hashtags:
                        if len(left_hts_in_profile) > 0:
                            cooccur_mat[0][hashtags.index(ht)] += 1
                        if len(right_hts_in_profile) > 0:
                            cooccur_mat[1][hashtags.index(ht)] += 1
                        ht_occur_vec[hashtags.index(ht)] += 1
        
        with open('data/from_anu/v1/{}_{}_hashtags_entropy.csv'.format(self.util.campaign, self.util.ea_type), mode='w') as file:
            fieldnames = ['Hashtag', '#occurences', '#left', '#right', 'Entropy']
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fieldnames)
            for i in range(len(hashtags)):
                if (cooccur_mat[0, i] + cooccur_mat[1, i]) > 0:
                    l_prob = float(cooccur_mat[0, i]) / (cooccur_mat[0, i] + cooccur_mat[1, i])
                    r_prob = float(cooccur_mat[1, i]) / (cooccur_mat[0, i] + cooccur_mat[1, i])
                    writer.writerow([hashtags[i], ht_occur_vec[i], cooccur_mat[0, i], cooccur_mat[1, i], entropy([l_prob, r_prob], base=2)])
        
        return hashtags, ht_occur_vec, cooccur_mat
        
    
    ## Assign the political leanings (left vs. right) to users as seed users.
    def assignUserLeaningLabels(self):
        available_users_ids = pickle.load(open(self.ea_users_ids_path, 'rb'))
        users = pickle.load(open(self.users_path, 'rb'))
        print(len(users.keys()))
        
        left_hts = []
        with open(self.left_extended_political_hashtags_path, 'r') as f:
            inp = f.readlines()
            left_hts = [ht.rstrip() for ht in inp]
        
        right_hts = []
        with open(self.right_extended_political_hashtags_path, 'r') as f:
            inp = f.readlines()
            right_hts = [ht.rstrip() for ht in inp]
        
        user_leanings = {}
        for user_id in users:
            ## left: 0, right = 1, unknown = -1 
            leaning = -1
            if user_id in available_users_ids:
                user_desc = users[user_id]['_source']['description']
                user_hashtags = list(set(re.findall(r"#(\w+)", user_desc)))
                user_hashtags = [tag.lower() for tag in user_hashtags]
                left_hts_in_profile = []
                right_hts_in_profile = []
                for ht in user_hashtags:
                    if ht in left_hts:
                        left_hts_in_profile.append(ht)
                    elif ht in right_hts:
                        right_hts_in_profile.append(ht)
                
                if (len(left_hts_in_profile) + len(right_hts_in_profile)) > 0:
                    if float(len(left_hts_in_profile)) / (len(left_hts_in_profile) + len(right_hts_in_profile)) >= 0.9:
                        leaning = 0
                    elif float(len(right_hts_in_profile)) / (len(left_hts_in_profile) + len(right_hts_in_profile)) >= 0.9:
                        leaning = 1

                user_leanings[user_id] = leaning
        
        pickle.dump(user_leanings, open(self.ea_seed_users_leanings_labels_path, 'wb'))
        return user_leanings
   
    
    ###########################################################################
    ## VIDEO Profile Enhancement operations ###################################
    ###########################################################################
    ## Get filtered video ids
    def getFilteredVideoIds(self):
        filtered_video_ids = pickle.load(open(self.videos_path, 'rb'))
        filtered_video_ids = list(filtered_video_ids.keys())
        print('len(filtered_video_ids):', len(filtered_video_ids))
        return filtered_video_ids
    
    ## Get filtered video ids
    def getVideoTweetsUsers(self, tweets):
        filtered_video_ids = self.getFilteredVideoIds()
        filtered_videos = {}
        for tid in tweets:
            uid = tweets[tid]['_source']['user_id_str']
            original_video_ids = tweets[tid]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tid]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tid]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            for video_id in video_ids:
                if video_id in filtered_video_ids:
                    if video_id not in filtered_videos:
                        filtered_videos[video_id] = {'uids': [uid], 'tids': [tid]}
                    else:
                        if uid not in filtered_videos[video_id]['uids']:
                            filtered_videos[video_id]['uids'].append(uid)
                        filtered_videos[video_id]['tids'].append(tid)
        return filtered_videos
    
    ## Assign the political leanings (left vs. right) to videos from users that promote them.
    def assignVideoLeaningLabels(self, tweets):
        users_inferred_leanings_scores = pickle.load(open(self.ea_users_inferred_leanings_scores_path, 'rb'))
        filtered_video_ids = self.getFilteredVideoIds()
        counter = 0
        ## assign leanings to videos.
        video_users_probs = {}
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            leaning_scores = users_inferred_leanings_scores[user_id]

            original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            
            for video_id in video_ids:
                if video_id in filtered_video_ids:
                    if (leaning_scores['left'] + leaning_scores['right']) != 0:
                        left_prob = float(leaning_scores['left']) / (leaning_scores['left'] + leaning_scores['right'])
                        right_prob = float(leaning_scores['right']) / (leaning_scores['left'] + leaning_scores['right'])
                        if video_id not in video_users_probs:
                            video_users_probs[video_id] = {}
                            if (leaning_scores['left'] + leaning_scores['right']) > 0:
                                video_users_probs[video_id][user_id] = {'left': left_prob, 'right': right_prob}
                        else:
                            if user_id not in video_users_probs[video_id]:
                                if (leaning_scores['left'] + leaning_scores['right']) > 0:
                                    video_users_probs[video_id][user_id] = {'left': left_prob, 'right': right_prob}
                    else:
                        counter+=1
                
                if video_id == 'Rw9E_nGVXxA':
                    print(user_id, leaning_scores)
        
        video_leanings_probs = {}
        for video_id in video_users_probs:
            left_probs = [video_users_probs[video_id][user_id]['left'] for user_id in video_users_probs[video_id]]
            right_probs = [video_users_probs[video_id][user_id]['right'] for user_id in video_users_probs[video_id]]
            avg_left_prob = float(sum(left_probs))/len(left_probs)
            avg_right_prob = float(sum(right_probs))/len(right_probs)
            video_leanings_probs[video_id] = {'left': avg_left_prob, 'right': avg_right_prob}
                
        print('Tweets by non-politic users: {}'.format(counter))
        return video_leanings_probs
    
    
    ## Return the tweets promoting the specific video by video-id.
    def getTweetsByVideoId(self, tweets, video_id):
        results = {}
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            #user_screen_name = users[user_id]['_source']['screen_name']
            original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            if video_id in video_ids:
                '''
                print(tweet_id)
                print("https://twitter.com/{}/status/{}".format(user_screen_name, tweet_id))
                print(tweets[tweet_id]['_source']['original_text'])
                if tweets[tweet_id]['_source']['retweeted_text'].strip() != 'N':
                    print(tweets[tweet_id]['_source']['retweeted_text'])
                if tweets[tweet_id]['_source']['quoted_text'].strip() != 'N':
                    print(tweets[tweet_id]['_source']['quoted_text'])
                print('--------------------------------------------------------')
                '''
                #results.append(tweets[tweet_id]['_source']['original_text'])
                #results.append(tweets[tweet_id])
                results[tweet_id] = tweets[tweet_id]
        
        return results
    
    ## Create the sub-following dict to make the calculation of tweet-cascade for a video computational-efficient.
    ## Run only once.
    def createSubFollowingsDictForCascade(self):
        if os.path.isfile(self.ea_users_sub_followings_path):
            print("sub_following_list already exists!! {}".format(self.ea_users_sub_followings_path))
        else:
            followers = pickle.load(open(self.ea_users_followers_path, 'rb'))
            tmp_followers = {}
            for uid in followers:
                tmp_followers[uid] = {}
                for fid in followers[uid]:
                    tmp_followers[uid][fid] = 0

            sub_followings_dict = {}

            cnt = 0
            for uid in followers:
                sub_followings_dict[uid] = {}
                for uid_2 in tmp_followers:
                    if uid != uid_2 and uid in tmp_followers[uid_2]:              
                        sub_followings_dict[uid][uid_2] = 0

                cnt+=1
                if cnt % 100 == 0:
                    print('User {} is handled!'.format(cnt))

            pickle.dump(sub_followings_dict, open(self.ea_users_sub_followings_path, 'wb'))
            print("sub_following_list created!! {}".format(self.ea_users_sub_followings_path))
    
    
    ## Find L, R, N videos
    def separateVideosByLeaning(self, video_leanings_probs):
        recfluence = pd.read_csv('data/media_bias/recfluence.csv')
        recfluence.dropna(subset=['CHANNEL_ID', 'LR'], inplace=True)
        recfluence = recfluence[['CHANNEL_ID', 'CHANNEL_TITLE', 'LR']]
        #print(recfluence.head())
        videos = pickle.load(open(self.videos_path, 'rb'))
        
        recfluence_channel_ids = set(recfluence['CHANNEL_ID'])
                
        cid2vid_dict = {}
        for vid in video_leanings_probs:
            cid = videos[vid]['_source']['snippet']['channelId']
            right_leaning_prob = video_leanings_probs[vid]['right']
            if cid not in cid2vid_dict:
                cid2vid_dict[cid] = []
                cid2vid_dict[cid].append(right_leaning_prob)
            else:
                cid2vid_dict[cid].append(right_leaning_prob)
        
        #topic
        print('#channels: {}, #videos: {}'.format(len(cid2vid_dict.keys()), sum([len(cid2vid_dict[cid]) for cid in cid2vid_dict])))
        #recfluence.intersect(topic)
        common_yt_channel_ids = recfluence_channel_ids.intersection(set(cid2vid_dict.keys()))
        print('#channels from rec. n dataset: {}, #videos from rec. n dataset: {}'.format(len(common_yt_channel_ids), sum([len(cid2vid_dict[cid]) for cid in common_yt_channel_ids])))

        
        inferred_leaning_probs = {'CHANNEL_ID': [], 'avg_cid_probs': [], 'num_videos_per_cid': []}
        for cid in common_yt_channel_ids:
            inferred_leaning_probs['CHANNEL_ID'].append(cid)
            inferred_leaning_probs['avg_cid_probs'].append(np.mean(cid2vid_dict[cid]))
            inferred_leaning_probs['num_videos_per_cid'].append(len(cid2vid_dict[cid]))
        inferred_leaning_probs = pd.DataFrame.from_dict(inferred_leaning_probs)
        
        df_union = recfluence.merge(inferred_leaning_probs, left_on='CHANNEL_ID', right_on='CHANNEL_ID')
        df_union.dropna(subset=['avg_cid_probs'], inplace=True)
        df_union = df_union.set_index('CHANNEL_ID')

        common_yt_video_ids = list(df_union.index)
        
        channel_political_ideologies = {}
        for cid in cid2vid_dict:
            gt_leaning = None
            if cid in common_yt_video_ids:
                #print(cid, df_union.loc[cid, 'CHANNEL_TITLE'], df_union.loc[cid, 'avg_cid_probs'], df_union.loc[cid, 'LR'])
                gt_leaning = str(df_union.loc[cid, 'LR'])
                channel_political_ideologies[cid] = gt_leaning
        print('#cids with ground_truth: {}'.format(len(channel_political_ideologies)))
        
        #vids = {'L': [], 'R': [], 'N': []}
        vids = {'L': {}, 'R': {}, 'N': {}}
        for vid in video_leanings_probs:
            cid = videos[vid]['_source']['snippet']['channelId']
            prob = video_leanings_probs[vid]['right']
            if cid in channel_political_ideologies:
                c_leaning = channel_political_ideologies[cid]
                if c_leaning == 'C':
                    #vids['N'].append(vid)
                    vids['N'][vid] = {'prob': prob, 'type': 'GT'}
                else:
                    #vids[channel_political_ideologies[cid]].append(vid)
                    vids[channel_political_ideologies[cid]][vid] = {'prob': prob, 'type': 'GT'}
            else:
                #right_leaning_prob = np.mean(cid2vid_dict[cid])
                if prob < self.predefined_video_leaning_thr['left']:
                    #vids['L'].append(vid)
                    vids['L'][vid] = {'prob': prob, 'type': 'inferred'}
                elif prob > self.predefined_video_leaning_thr['right']:
                    #vids['R'].append(vid)
                    vids['R'][vid] = {'prob': prob, 'type': 'inferred'}
                else:
                    #vids['N'].append(vid)
                    vids['N'][vid] = {'prob': prob, 'type': 'inferred'}
        
        #pickle.dump(vids, open('{}_{}_video_leanings.pkl'.format(self.util.campaign, self.util.ea_type), 'wb'))
        #return vids
        
        vids_new = {'L':[], 'R': [], 'N': []}
        for leaning in vids:
            for vid in vids[leaning]:
                vids_new[leaning].append(vid)
        
        return vids_new
                
    ## CASCADE MEASURES  ########################################################################
    ## Return the tweet cascade of a given video by id
    def getTweetCascadeByVideoId(self, tweets, video_id):
        related_tweets = self.getTweetsByVideoId(tweets, video_id)
        sub_followings_dict = pickle.load(open(self.ea_users_sub_followings_path, 'rb'))
        
        user_time_dict = {}
        user_tweet_type_dict = {}
        for tid in related_tweets:
            user_id = tweets[tid]['_source']['user_id_str']
            retweeted_tweet_id_str = tweets[tid]['_source']['retweeted_tweet_id_str']
            quoted_tweet_id_str = tweets[tid]['_source']['quoted_tweet_id_str']
            reply_user_id_str = tweets[tid]['_source']['reply_user_id_str']
            original_video_ids = tweets[tid]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tid]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tid]['_source']['quoted_vids'].split(';')
            if user_id not in user_time_dict:
                user_time_dict[user_id] = int(tweets[tid]['_source']['timestamp_ms'])
                if retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N':
                    user_tweet_type_dict[user_id] = 'retweet'
                elif retweeted_tweet_id_str == 'N' and quoted_tweet_id_str != 'N' and video_id in quoted_video_ids:
                    user_tweet_type_dict[user_id] = 'quoted'
                elif retweeted_tweet_id_str == 'N' and quoted_tweet_id_str == 'N':
                    user_tweet_type_dict[user_id] = 'original'
                else:
                    user_tweet_type_dict[user_id] = 'original'
                if reply_user_id_str != 'N':
                    user_tweet_type_dict[user_id] = 'reply'
            else:
                if int(related_tweets[tid]['_source']['timestamp_ms']) < user_time_dict[user_id]:
                    user_time_dict[user_id] = int(tweets[tid]['_source']['timestamp_ms'])
                    if retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N':
                        user_tweet_type_dict[user_id] = 'retweet'
                    elif retweeted_tweet_id_str == 'N' and quoted_tweet_id_str != 'N' and video_id in quoted_video_ids:
                        user_tweet_type_dict[user_id] = 'quoted'
                    elif retweeted_tweet_id_str == 'N' and quoted_tweet_id_str == 'N':
                        user_tweet_type_dict[user_id] = 'original'
                    else:
                        user_tweet_type_dict[user_id] = 'original'
                    if reply_user_id_str != 'N':
                        user_tweet_type_dict[user_id] = 'reply'
        
        print("-- Users sorted by timestamp_ms! #of users: {}".format(len(user_time_dict.keys())))
        
        user_ids = [item[0] for item in sorted(user_time_dict.items(), key=itemgetter(1))]
        
        related_followings_dict = {}
        for uid in user_ids:
            related_followings_dict[uid] = copy.deepcopy(sub_followings_dict[uid])
        
        all_cascade_list = {}
        users_immediate_neighbors = {}
        for i in range(len(user_ids)):
            all_cascade_list[i] = {}
            users_immediate_neighbors[i] = []
        
        for i in range(len(user_ids)):
            if i % 50 == 0:
                print("-- User {} is processed!".format(str(i)))
            
            followers_uids = related_followings_dict[user_ids[i]]
            followers = [user_ids.index(uid) for uid in followers_uids if uid in user_ids and user_ids.index(uid) < i]
            
            min_cascades = []
            max_cascades = []
            for fol_ind in followers:
                min_cascades.append(all_cascade_list[fol_ind]['min'])
                max_cascades.append(all_cascade_list[fol_ind]['max'])
            
            if len(followers) == 0:
                all_cascade_list[i]['min'] = 1
                all_cascade_list[i]['max'] = 1
            else:
                all_cascade_list[i]['min'] = min(min_cascades) + 1
                all_cascade_list[i]['max'] = max(max_cascades) + 1
                
                users_immediate_neighbors[i].append(followers[min_cascades.index(min(min_cascades))])
        
        cascades = {}
        for idx in all_cascade_list:
            cascades[user_ids[idx]] = all_cascade_list[idx]
        
        return user_ids, cascades, users_immediate_neighbors
    
    ## Calcuate tweet cascade for all filtered videos.
    def getAllTweetCascades(self, tweets):
        cascades = {}
        if os.path.isfile(self.ea_cascade_measures_path):
            cascades = pickle.load(open(self.ea_cascade_measures_path, 'rb'))
        else:
            video_ids = self.getFilteredVideoIds()
            cnt = 0
            for vid in video_ids:
                print('Cascade calculating for {} ... | cnt: {}'.format(vid, cnt))
                _, cascade, _ = self.getTweetCascadeByVideoId(tweets, vid)
                cascades[vid] = cascade
                print("Cascade calcualted for {} !".format(vid))
                cnt+=1
            
            pickle.dump(cascades, open(self.ea_cascade_measures_path, 'wb'))
            print("Cascades saved !! {}".format(self.ea_cascade_measures_path))
        
        return cascades
    
    ## Analyze cascade measures
    def analyzeCascadeMeasures(self, video_leanings_probs):
        cascades = pickle.load(open(self.ea_cascade_measures_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        mean_cascades = {}
        median_cascades = {}
        max_cascadess = {}
        num_sources = {}
        for vid in video_leanings_probs:
            mean_cascades[vid] = np.mean([cascades[vid][uid]['min'] for uid in cascades[vid]])
            median_cascades[vid] = np.median([cascades[vid][uid]['min'] for uid in cascades[vid]])
            max_cascadess[vid] = np.amax([cascades[vid][uid]['min'] for uid in cascades[vid]])
            num_sources[vid] = len([cascades[vid][uid]['min'] for uid in cascades[vid] if cascades[vid][uid]['min']==1])
        
        mean_cascades_left = [mean_cascades[vid] for vid in vids['L']]
        mean_cascades_right = [mean_cascades[vid] for vid in vids['R']]
        mean_cascades_neutral = [mean_cascades[vid] for vid in vids['N']]
        median_cascades_left = [median_cascades[vid] for vid in vids['L']]
        median_cascades_right = [median_cascades[vid] for vid in vids['R']]
        median_cascades_neutral = [median_cascades[vid] for vid in vids['N']]
        max_cascades_left = [max_cascadess[vid] for vid in vids['L']]
        max_cascades_right = [max_cascadess[vid] for vid in vids['R']]
        max_cascades_neutral = [max_cascadess[vid] for vid in vids['N']]
        num_sources_left = [num_sources[vid] for vid in vids['L']]
        num_sources_right = [num_sources[vid] for vid in vids['R']]
        num_sources_neutral = [num_sources[vid] for vid in vids['N']]
        
        print('#left videos: {}'.format(len(mean_cascades_left)))
        print('#right videos: {}'.format(len(mean_cascades_right)))
        print('#neutral videos: {}'.format(len(mean_cascades_neutral)))

        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([mean_cascades_left, mean_cascades_right, mean_cascades_neutral], 
                             ["Left", "Right", "Neutral"], "mean_min_cascades", None, False)
        self.util.plotCDFMultiple([median_cascades_left, median_cascades_right, median_cascades_neutral], 
                             ["Left", "Right", "Neutral"], "median_min_cascades", None, False)
        self.util.plotCDFMultiple([max_cascades_left, max_cascades_right, max_cascades_neutral], 
                             ["Left", "Right", "Neutral"], "max_min_cascades", None, False)
        self.util.plotCDFMultiple([num_sources_left, num_sources_right, num_sources_neutral], 
                             ["Left", "Right", "Neutral"], "#sources", None, False)

        self.util.plotHistKDEMulitple([mean_cascades_left, mean_cascades_right, mean_cascades_neutral], 
                                 ["Left", "Right", "Neutral"], "mean_min_cascades", None, False)
        self.util.plotHistKDEMulitple([median_cascades_left, median_cascades_right, median_cascades_neutral], 
                                 ["Left", "Right", "Neutral"], "median_min_cascades", None, False)
        self.util.plotHistKDEMulitple([max_cascades_left, max_cascades_right, max_cascades_neutral], 
                                 ["Left", "Right", "Neutral"], "max_min_cascades", None, False)
        self.util.plotHistKDEMulitple([num_sources_left, num_sources_right, num_sources_neutral], 
                                 ["Left", "Right", "Neutral"], "#sources", None, False)

        ## Significance test
        print("Mean Cascade Significant Test")
        self.util.applySignificanceTest(mean_cascades_left, mean_cascades_right)
        print("Median Cascade Significant Test")
        self.util.applySignificanceTest(median_cascades_left, median_cascades_right)
        print("Max Cascade (Depth) Significant Test")
        self.util.applySignificanceTest(max_cascades_left, max_cascades_right)
        print("Number of sources Significant Test")
        self.util.applySignificanceTest(num_sources_left, num_sources_right)
        
    
    ## NW STRUCTURAL MEASURES  ####################################################################################
    ## Calculate how early early adopters promote the video in terms of seconds.
    ## Returns video_promotion_delta = {vid_1: {uid_1: seconds, uid_2: seconds, ...}, ...}
    def getNetworkStructureMeasuresByVideoId(self, tweets, video_id, users_num_followers, users_leaning_probs, users_leaning_labels):
        related_tweets = self.getTweetsByVideoId(tweets, video_id)
        sub_followings_dict = pickle.load(open(self.ea_users_sub_followings_path, 'rb'))
        
        users_num_tweets = {}
        video_tweets_info = []
        for tid in related_tweets:
            tweet_id = tweets[tid]['_source']['tweet_id_str']
            user_id = tweets[tid]['_source']['user_id_str']
            timestamp_ms = int(tweets[tid]['_source']['timestamp_ms'])
            video_tweets_info.append((user_id, tweet_id, timestamp_ms))
            if user_id not in users_num_tweets:
                users_num_tweets[user_id] = 1
            else:
                users_num_tweets[user_id] += 1
        
        video_tweets_info_sorted = sorted(video_tweets_info, key=lambda tup: tup[2])        
        
        user_ids = []
        for tup in video_tweets_info_sorted:
            if tup[0] not in user_ids:
                user_ids.append(tup[0])
        
        nw_attributes = {}
        for uid in user_ids:
            nw_attributes[uid] = {'num_tweets': users_num_tweets[uid], 
                                  'num_followers': users_num_followers[uid]}
            if uid in users_leaning_probs:
                nw_attributes[uid]['leaning_probs_raw'] = int(round(users_leaning_probs[uid]*10, 0))
                nw_attributes[uid]['leaning_probs_abs'] = int(round(abs(users_leaning_probs[uid]-0.5)*10, 0))
                nw_attributes[uid]['leaning_labels'] = users_leaning_labels[uid]
        
        ## create follower-followee graph and set attributes
        G = nx.DiGraph()
        G.add_nodes_from(user_ids)
        print('len(G): {}'.format(len(G)))
        for uid in user_ids:
            for uid_2 in sub_followings_dict[uid]:
                if uid_2 in user_ids:
                    G.add_edge(uid, uid_2)
        
        nx.set_node_attributes(G, nw_attributes)
        
        #print('len(user_ids): {}'.format(len(user_ids)))
        #print('len(nw_attributes.keys()): {}'.format(len(nw_attributes.keys())))
        #print('len(G): {}'.format(len(G)))
        
        ## Start caluclating Structure measures
        
        # calcuate network_size
        nw_size = len(G)
        
        # calcuate largest indegree
        nw_max_indegree = max(list(dict(G.in_degree).values()))
        
        # calcluate density of the network
        nw_density = nx.classes.function.density(G)
        print('---Density calculated')
        
        # calculate centrality measures
        in_degree_centrality = nx.algorithms.centrality.in_degree_centrality(G)
        nw_in_degree_centrality_gini = self.util.gini(list(in_degree_centrality.values()))
        nw_in_degree_centrality_mean = np.mean(list(in_degree_centrality.values()))
        nw_in_degree_centrality_median = np.median(list(in_degree_centrality.values()))
        out_degree_centrality = nx.algorithms.centrality.out_degree_centrality(G)
        nw_out_degree_centrality_mean = np.mean(list(out_degree_centrality.values()))
        nw_out_degree_centrality_median = np.median(list(out_degree_centrality.values()))
        nw_out_degree_centrality_gini = self.util.gini(list(out_degree_centrality.values()))
        print('---Degree centrality calculated')
        closeness_centralities = nx.algorithms.closeness_centrality(G, wf_improved=True)
        nw_closeness_centrality_max = max(closeness_centralities.values())
        nw_closeness_centrality_gini = self.util.gini(list(closeness_centralities.values()))
        nw_closeness_centrality_mean = np.mean(list(closeness_centralities.values()))
        nw_closeness_centrality_median = np.median(list(closeness_centralities.values()))
        print('---Closeness centrality calculated')
        betweenness_centralities = nx.algorithms.betweenness_centrality(G, normalized=True, endpoints=False)
        nw_betweenness_centrality_max = max(betweenness_centralities.values())
        nw_betweenness_centrality_gini = self.util.gini(list(betweenness_centralities.values()))
        nw_betweenness_centrality_mean = np.mean(list(betweenness_centralities.values()))
        nw_betweenness_centrality_median = np.median(list(betweenness_centralities.values()))
        print('---Betweenness centrality calculated')
        #print(list(closeness_centralities.values()), nw_closeness_centrality_gini)
        #print(list(betweenness_centralities.values()), nw_betweenness_centrality_gini)
        
        # calculate global efficiency
        nw_global_efficiency = graph_ops.calculateGlobalEfficiency(G) 
        print('---Global Efficiency calculated')
        
        nw_degree_assortativity = None
        nw_assortativity_leaning_probs_raw = None
        nw_assortativity_leaning_probs_abs_nw1 = None
        nw_assortativity_leaning_probs_abs_nw2 = None
        nw_assortativity_leaning_labels = None
        nw_assortativity_num_tweets = None
        nw_assortativity_num_followers = None
        # calculate degree assortitavity
        try:
            nw_degree_assortativity = nx.algorithms.assortativity.degree_assortativity_coefficient(G, x='in', y='in')
            #nw_degree_assortativity = graph_ops.calculateDegreeAssortativity(G, 'in', 'in')
            print('---Assortitavity (degree) calculated')
        except Exception as e:
            #print(e)
            pass
        # calculate assortitavity w.r.t. num_tweets
        try:
            nw_assortativity_num_tweets = nx.algorithms.assortativity.numeric_assortativity_coefficient(G, 'num_tweets')
            #nw_assortativity_num_tweets = graph_ops.calculateNumericAssortativity(G, 'num_tweets')
            print('---Assortitavity (w.r.t. num_tweets) calculated')
        except Exception as e:
            #print(e)
            pass
        '''
        # calculate assortitavity w.r.t. num_followers
        try:
            nw_assortativity_num_followers = nx.algorithms.assortativity.numeric_assortativity_coefficient(G, 'num_followers')
            print('---Assortitavity (w.r.t. num_followers) calculated')
        except Exception as e:
            #print(e)
            pass
        '''
        # calculate assortitavity w.r.t. leaning
        try:
            users_to_be_removed = list(set(user_ids).difference(set(users_leaning_probs.keys())))
            G.remove_nodes_from(users_to_be_removed)
            nw_assortativity_leaning_probs_raw = nx.algorithms.assortativity.numeric_assortativity_coefficient(G, 'leaning_probs_raw')
            nw_assortativity_leaning_probs_abs_nw1 = nx.algorithms.assortativity.numeric_assortativity_coefficient(G, 'leaning_probs_abs')
            nw_assortativity_leaning_labels = nx.algorithms.assortativity.attribute_assortativity_coefficient(G, 'leaning_labels')
            G2 = G.to_undirected(reciprocal=True)
            nw_assortativity_leaning_probs_abs_nw2 = nx.algorithms.assortativity.numeric_assortativity_coefficient(G2, 'leaning_probs_abs')
            #nw_assortativity_leaning = graph_ops.calculateNumericAssortativity(G, 'leaning')
            print('---Assortitavity (w.r.t. leaning) calculated')
        except Exception as e:
            #print(e)
            pass
        #print(nw_degree_assortativity, nw_assortativity_leaning, nw_assortativity_num_tweets, nw_assortativity_num_followers)
        #print(G.nodes)
        #print(G.edges)
        '''
        if nw_degree_assortativity != None and math.isnan(nw_degree_assortativity):
            print(G.in_degree)
            print(G.edges)
        '''
        
        return nw_size, nw_max_indegree, nw_density, nw_in_degree_centrality_gini, nw_in_degree_centrality_mean, nw_in_degree_centrality_median, nw_out_degree_centrality_gini, nw_out_degree_centrality_mean, nw_out_degree_centrality_median, nw_closeness_centrality_max, nw_closeness_centrality_gini, nw_closeness_centrality_mean, nw_closeness_centrality_median, nw_betweenness_centrality_max, nw_betweenness_centrality_gini, nw_betweenness_centrality_mean, nw_betweenness_centrality_median, nw_global_efficiency, nw_degree_assortativity, nw_assortativity_leaning_probs_raw, nw_assortativity_leaning_probs_abs_nw1, nw_assortativity_leaning_probs_abs_nw2, nw_assortativity_leaning_labels, nw_assortativity_num_tweets, nw_assortativity_num_followers
    
    
    ## Calcuate structural measures for all filtered videos.
    def getAllNetworkStructureMeasures(self, tweets):
        users = pickle.load(open(self.users_path, 'rb'))
        users_num_followers = {}
        for uid in users:
            users_num_followers[uid] = int(users[uid]['_source']['followers_count'])
        users_leaning_scores = pickle.load(open(self.ea_users_inferred_leanings_scores_path, 'rb'))
        users_leaning_probs = {}
        users_leaning_labels = {}
        for uid in users_leaning_scores:
            if (users_leaning_scores[uid]['left'] + users_leaning_scores[uid]['right']) != 0:
                user_leaning_prob = users_leaning_scores[uid]['right'] / (users_leaning_scores[uid]['left'] + users_leaning_scores[uid]['right'])
                users_leaning_probs[uid] = user_leaning_prob
                #users_leaning_probs_raw[uid] = int(round(user_leaning_prob*10, 0))
                #users_leaning_probs_abs[uid] = int(round(abs(user_leaning_prob-0.5)*10, 0))
                
                if user_leaning_prob > self.predefined_video_leaning_thr['right']:
                    users_leaning_labels[uid] = 'R'
                elif user_leaning_prob < self.predefined_video_leaning_thr['left']:
                    users_leaning_labels[uid] = 'L'
                else:
                    users_leaning_labels[uid] = 'N'
        
        structural_measures = {}
        if os.path.isfile(self.ea_network_structural_measures_path):
            structural_measures = pickle.load(open(self.ea_network_structural_measures_path, 'rb'))
        else:
            video_ids = self.getFilteredVideoIds()
            cnt = 0
            for vid in video_ids:
                print('Structural measures calculating for {} ... | cnt: {}'.format(vid, cnt))
                nw_size, nw_max_indegree, nw_density, nw_in_degree_centrality_gini, nw_in_degree_centrality_mean, nw_in_degree_centrality_median, nw_out_degree_centrality_gini, nw_out_degree_centrality_mean, nw_out_degree_centrality_median, nw_closeness_centrality_max, nw_closeness_centrality_gini, nw_closeness_centrality_mean, nw_closeness_centrality_median, nw_betweenness_centrality_max, nw_betweenness_centrality_gini, nw_betweenness_centrality_mean, nw_betweenness_centrality_median, nw_global_efficiency, nw_degree_assortativity, nw_assortativity_leaning_probs_raw, nw_assortativity_leaning_probs_abs_nw1, nw_assortativity_leaning_probs_abs_nw2, nw_assortativity_leaning_labels,  nw_assortativity_num_tweets, nw_assortativity_num_followers = self.getNetworkStructureMeasuresByVideoId(tweets, vid, users_num_followers, users_leaning_probs, users_leaning_labels)
                structural_measures[vid] = {"nw_size": nw_size, 
                                            "nw_max_indegree": nw_max_indegree, 
                                            "nw_density": nw_density,
                                            "nw_in_degree_centrality_gini": nw_in_degree_centrality_gini, 
                                            "nw_in_degree_centrality_mean": nw_in_degree_centrality_mean, 
                                            "nw_in_degree_centrality_median": nw_in_degree_centrality_median, 
                                            "nw_out_degree_centrality_gini": nw_out_degree_centrality_gini, 
                                            "nw_out_degree_centrality_mean": nw_out_degree_centrality_mean, 
                                            "nw_out_degree_centrality_median": nw_out_degree_centrality_median,
                                            "nw_closeness_centrality_max": nw_closeness_centrality_max, 
                                            "nw_closeness_centrality_gini": nw_closeness_centrality_gini, 
                                            "nw_closeness_centrality_mean": nw_closeness_centrality_mean, 
                                            "nw_closeness_centrality_median": nw_closeness_centrality_median, 
                                            "nw_betweenness_centrality_max": nw_betweenness_centrality_max, 
                                            "nw_betweenness_centrality_gini": nw_betweenness_centrality_gini, 
                                            "nw_betweenness_centrality_mean": nw_betweenness_centrality_mean, 
                                            "nw_betweenness_centrality_median": nw_betweenness_centrality_median, 
                                            "global_efficiency": nw_global_efficiency, 
                                            "nw_degree_assortativity": nw_degree_assortativity, 
                                            "nw_assortativity_leaning_probs_raw": nw_assortativity_leaning_probs_raw, 
                                            "nw_assortativity_leaning_probs_abs_nw1": nw_assortativity_leaning_probs_abs_nw1, 
                                            "nw_assortativity_leaning_probs_abs_nw2": nw_assortativity_leaning_probs_abs_nw2, 
                                            "nw_assortativity_leaning_labels": nw_assortativity_leaning_labels, 
                                            "nw_assortativity_num_tweets": nw_assortativity_num_tweets, 
                                            "nw_assortativity_num_followers": nw_assortativity_num_followers}
                #print("Structural measures calcualted for {} !".format(vid))
                cnt+=1
            
            pickle.dump(structural_measures, open(self.ea_network_structural_measures_path, 'wb'))
            print("Structural measures saved !! {}".format(self.ea_network_structural_measures_path))
        
        return structural_measures
    
    ## Analyze Structural measures
    def analyzeStructuralMeasures(self, measure_type, video_leanings_probs):
        structural_measures = {}
        structural_measures = pickle.load(open(self.ea_network_structural_measures_path, 'rb'))
        
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        measures_left = [structural_measures[vid][measure_type] for vid in vids['L'] if (structural_measures[vid][measure_type] != None and not math.isnan(structural_measures[vid][measure_type]) and not math.isinf(structural_measures[vid][measure_type]))]
        measures_right = [structural_measures[vid][measure_type] for vid in vids['R'] if (structural_measures[vid][measure_type] != None and not math.isnan(structural_measures[vid][measure_type]) and not math.isinf(structural_measures[vid][measure_type]))]
        measures_neutral = [structural_measures[vid][measure_type] for vid in vids['N'] if (structural_measures[vid][measure_type] != None and not math.isnan(structural_measures[vid][measure_type]) and not math.isinf(structural_measures[vid][measure_type]))]
        
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance test
        self.util.applySignificanceTest(measures_left, measures_right)
        
        measures_left = np.array(measures_left, dtype='float64')
        measures_right = np.array(measures_right, dtype='float64')
        measures_neutral = np.array(measures_neutral, dtype='float64')
        
        if measure_type == 'nw_max_indegree':
            measures_left[measures_left==0] = 0.1
            measures_right[measures_right==0] = 0.1
            measures_neutral[measures_neutral==0] = 0.1
        
        is_log = False
        if measure_type == 'nw_size' or measure_type == 'nw_max_indegree':
            is_log = True
        
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"], measure_type, None, is_log)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], measure_type, None, is_log)
    
    ## TEMPORAL MEASURES  ###################################################################################
    def getTemporalMeasuresByVideoId(self, tweets, video_id):
        sub_followings_dict = pickle.load(open(self.ea_users_sub_followings_path, 'rb'))
        related_tweets = self.getTweetsByVideoId(tweets, video_id)
        
        video_tweets_info = []
        for tid in related_tweets:
            tweet_id = tweets[tid]['_source']['tweet_id_str']
            user_id = tweets[tid]['_source']['user_id_str']
            timestamp_ms = int(tweets[tid]['_source']['timestamp_ms'])
            video_tweets_info.append((user_id, tweet_id, timestamp_ms))
        
        video_tweets_info_sorted = sorted(video_tweets_info, key=lambda tup: tup[2])        
        
        user_ids = []
        timestamps = []
        timestamps_first_tweets = []
        for tup in video_tweets_info_sorted:
            if tup[0] not in user_ids:
                #timestamps.append(tup[2])
                user_ids.append(tup[0])
                timestamps_first_tweets.append(tup[2])
            timestamps.append(tup[2])
        
        ## create follower-followee and temporal follower_followee graphs
        G_full = nx.DiGraph()
        G_temporal = nx.DiGraph()
        G_full.add_nodes_from(user_ids)
        G_temporal.add_nodes_from(user_ids)
        for uid in user_ids:
            for uid_2 in sub_followings_dict[uid]:
                if uid_2 in user_ids:
                    G_full.add_edge(uid, uid_2)
                if uid_2 in user_ids and user_ids.index(uid_2) < user_ids.index(uid):
                    G_temporal.add_edge(uid, uid_2)
        
        ## Start caluclating Structure measures
        ts_diffs_to_first_tweet = []
        for i in range(1, len(timestamps)):
            diff = float((timestamps[i] - timestamps[0])) / (60 * 1000)
            ts_diffs_to_first_tweet.append(diff)
            if diff < 0.:
                print(vid, 'There is a problem')
        
        # calculate mean time difference w.r.t the first tweet
        nw_temporal_diff_wrt_first_tweet_mean = np.mean(ts_diffs_to_first_tweet)
        # calculate median time difference w.r.t the first tweet
        nw_temporal_diff_wrt_first_tweet_median = np.median(ts_diffs_to_first_tweet)
        
        ts_diffs_betweetn_tweets = []
        for i in range(1, len(timestamps)):
            diff = float((timestamps[i] - timestamps[i-1])) / (60 * 1000)
            ts_diffs_betweetn_tweets.append(diff)
            if diff < 0.:
                print(vid, 'There is a problem')
        
        # calculate mean time difference between tweets
        nw_temporal_diff_between_pairs_mean = np.mean(ts_diffs_betweetn_tweets)
        # calculate median time difference between tweets
        nw_diff_speed_mnw_temporal_diff_between_pairs_median = np.median(ts_diffs_betweetn_tweets)
        
        # calculate life time in early adoption period
        nw_life_time = float(timestamps[-1] - timestamps[0]) / (60 * 1000)
        
        # calculate average time diff between the first tweets of source users.
        tmp_ts = []
        for i in range(len(user_ids)):
            if G_temporal.out_degree[user_ids[i]] == 0:
                tmp_ts.append(timestamps_first_tweets[i])
        ts_diffs = []
        for i in range(1, len(tmp_ts)):
            diff = float((tmp_ts[i] - tmp_ts[i-1])) / (60 * 1000)
            ts_diffs.append(diff)
        nw_temporal_diff_between_first_tweets_of_source_users_mean = np.mean(ts_diffs)
        nw_temporal_diff_between_first_tweets_of_source_users_median = np.median(ts_diffs)
        
        # calculate the time diff between first tweet and first tweet of max indegree user
        indegrees = dict(G_full.in_degree)
        sorted_indegrees = sorted(indegrees.items(), key=itemgetter(1), reverse=True)
        uid_with_max_indegree = sorted_indegrees[0][0]
        nw_temporal_diff_between_max_indegree_user = float(timestamps_first_tweets[user_ids.index(uid_with_max_indegree)] - timestamps[0]) / (60 * 1000)
        
        #print(G.nodes)
        #print(G.edges)
        
        return nw_temporal_diff_wrt_first_tweet_mean, nw_temporal_diff_wrt_first_tweet_median, nw_temporal_diff_between_pairs_mean, nw_diff_speed_mnw_temporal_diff_between_pairs_median, nw_life_time, nw_temporal_diff_between_first_tweets_of_source_users_mean, nw_temporal_diff_between_first_tweets_of_source_users_median, nw_temporal_diff_between_max_indegree_user
    
    ## Calcuate temporal measures for all filtered videos.
    def getAllTemporalMeasures(self, tweets):
        temporal_measures = {}
        if os.path.isfile(self.ea_temporal_measures_path):
            temporal_measures = pickle.load(open(self.ea_temporal_measures_path, 'rb'))
        else:
            video_ids = self.getFilteredVideoIds()
            cnt = 0
            for vid in video_ids:
                print('Temporal measures calculating for {} ... | cnt: {}'.format(vid, cnt))
                nw_temporal_diff_wrt_first_tweet_mean, nw_temporal_diff_wrt_first_tweet_median, nw_temporal_diff_between_pairs_mean, nw_diff_speed_mnw_temporal_diff_between_pairs_median, nw_life_time, nw_temporal_diff_between_first_tweets_of_source_users_mean, nw_temporal_diff_between_first_tweets_of_source_users_median, nw_temporal_diff_between_max_indegree_user = self.getTemporalMeasuresByVideoId(tweets, vid)
                temporal_measures[vid] = {"nw_temporal_diff_wrt_first_tweet_mean": nw_temporal_diff_wrt_first_tweet_mean, 
                                       "nw_temporal_diff_wrt_first_tweet_median": nw_temporal_diff_wrt_first_tweet_median, 
                                       "nw_temporal_diff_between_pairs_mean": nw_temporal_diff_between_pairs_mean, 
                                       "nw_diff_speed_mnw_temporal_diff_between_pairs_median": nw_diff_speed_mnw_temporal_diff_between_pairs_median, 
                                       "nw_life_time": nw_life_time, 
                                       "nw_temporal_diff_between_first_tweets_of_source_users_mean": nw_temporal_diff_between_first_tweets_of_source_users_mean, 
                                       "nw_temporal_diff_between_first_tweets_of_source_users_median": nw_temporal_diff_between_first_tweets_of_source_users_median, 
                                       "nw_temporal_diff_between_max_indegree_user": nw_temporal_diff_between_max_indegree_user}
                #print("Temporal measures calcualted for {} !".format(vid))
                cnt+=1
            
            pickle.dump(temporal_measures, open(self.ea_temporal_measures_path, 'wb'))
            print("Temporal measures saved !! {}".format(self.ea_temporal_measures_path))
        
        return temporal_measures
    
    ## Analyze Temporal measures
    def analyzeTemporalMeasures(self, measure_type, video_leanings_probs):
        temporal_measures = pickle.load(open(self.ea_temporal_measures_path, 'rb'))
        
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        measures_left = [temporal_measures[vid][measure_type] for vid in vids['L']]
        measures_right = [temporal_measures[vid][measure_type] for vid in vids['R']]
        measures_neutral = [temporal_measures[vid][measure_type] for vid in vids['N']]
        
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance tests
        self.util.applySignificanceTest(measures_left, measures_right)
        
        measures_left = np.array(measures_left, dtype='float64')
        measures_right = np.array(measures_right, dtype='float64')
        measures_neutral = np.array(measures_neutral, dtype='float64')
        
        if measure_type == "nw_temporal_diff_between_max_indegree_user":
            measures_left[measures_left==0] = 0.1
            measures_right[measures_right==0] = 0.1
            measures_neutral[measures_neutral==0] = 0.1
        
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"],'{} in seconds'.format(measure_type), None, True)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], '{} in seconds'.format(measure_type), None, True)
    
    
    ## ENGAGEMENT MEASURES  ##########################################################################################
    def getEngagementMeasuresByVideoId(self, tweets, video_id):
        related_tweets = self.getTweetsByVideoId(tweets, video_id)
        users = []
        num_tweets = 0
        num_original_tweets = 0
        num_retweets = 0
        num_quoted_tweets = 0
        num_replies = 0
        for tid in related_tweets:
            uid = tweets[tid]['_source']['user_id_str']
            tweet_type = None
            retweeted_tweet_id_str = tweets[tid]['_source']['retweeted_tweet_id_str']
            quoted_tweet_id_str = tweets[tid]['_source']['quoted_tweet_id_str']
            reply_user_id_str = tweets[tid]['_source']['reply_user_id_str']

            if retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N':
                tweet_type = 'retweet'
            elif (retweeted_tweet_id_str == None or retweeted_tweet_id_str == 'N') and (quoted_tweet_id_str != None and quoted_tweet_id_str != 'N'):
                tweet_type = 'quoted'
            elif reply_user_id_str != None and reply_user_id_str != 'N':
                tweet_type = 'reply'
            else:
                tweet_type = 'original'
            
            users.append(uid)
            num_tweets += 1
            if tweet_type == 'original':
                num_original_tweets += 1
            elif tweet_type == 'retweet':
                num_retweets += 1
            elif tweet_type == 'quoted':
                num_quoted_tweets += 1
            elif tweet_type == 'reply':
                num_replies += 1
        
        return len(set(users)), num_tweets, num_original_tweets, num_retweets, num_quoted_tweets, num_replies 

    def getAllEngagementMeasures(self, tweets):
        engagement_measures = {}
        if os.path.isfile(self.ea_engagement_measures_path):
            engagement_measures = pickle.load(open(self.ea_engagement_measures_path, 'rb'))
        else:
            video_ids = self.getFilteredVideoIds()
            cnt = 0
            for vid in video_ids:
                print('Engagement measures calculating for {} ... | cnt: {}'.format(vid, cnt))
                num_users, num_tweets, num_original_tweets, num_retweets, num_quoted_tweets, num_replies = self.getEngagementMeasuresByVideoId(tweets, vid)
                engagement_measures[vid] = {"num_users": num_users, 
                                            "num_tweets": num_tweets, 
                                            "num_original_tweets": num_original_tweets, 
                                            "num_retweets": num_retweets, 
                                            "num_quoted_tweets": num_quoted_tweets, 
                                            "num_replies": num_replies}
                #print("Engagement measures calcualted for {} !".format(vid))
                cnt+=1
            
            pickle.dump(engagement_measures, open(self.ea_engagement_measures_path, 'wb'))
            print("Engagement measures saved !! {}".format(self.ea_engagement_measures_path))
        
        return engagement_measures
    
    ## Analyze Engagement measures
    def analyzeEngagementMeasures(self, measure_type, video_leanings_probs):
        engagement_measures = pickle.load(open(self.ea_engagement_measures_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        measures_left = [engagement_measures[vid][measure_type] for vid in vids['L']]
        measures_right = [engagement_measures[vid][measure_type] for vid in vids['R']]
        measures_neutral = [engagement_measures[vid][measure_type] for vid in vids['N']]
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance tests
        self.util.applySignificanceTest(measures_left, measures_right)
        
        measures_left = np.array(measures_left, dtype='float64')
        measures_right = np.array(measures_right, dtype='float64')
        measures_neutral = np.array(measures_neutral, dtype='float64')
        
        measures_left[measures_left==0] = 0.1
        measures_right[measures_right==0] = 0.1
        measures_neutral[measures_neutral==0] = 0.1
        
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"], measure_type, None, True)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], measure_type, None, True)
    
    
    ## LANGUAGE MEASURES  ########################################################################################
    ## Calcuate Language measures for all filtered videos.
    def getAllLanguageMeasures(self, tweets):
        language_liwc_measures = {}
        language_empath_measures = {}
        if os.path.isfile(self.ea_language_liwc_measures_path):
            language_liwc_measures = pickle.load(open(self.ea_language_liwc_measures_path, 'rb'))
            language_empath_measures = pickle.load(open(self.ea_language_empath_measures_path, 'rb'))
        else:
            filtered_video_ids = self.getFilteredVideoIds()
            parser = EkphrasisParser()
            cat, dic = liwc.read_liwc('src/language/liwc_data/LIWC2007_English131104.dic')
            empath_dic = Empath()
            counter = 0
            for tid in tweets:
                retweeted_tweet_id_str = tweets[tid]['_source']['retweeted_tweet_id_str']
                quoted_tweet_id_str = tweets[tid]['_source']['quoted_tweet_id_str']
                reply_user_id_str = tweets[tid]['_source']['reply_user_id_str']
                original_video_ids = tweets[tid]['_source']['original_vids'].split(';')
                retweeted_video_ids = tweets[tid]['_source']['retweeted_vids'].split(';')
                quoted_video_ids = tweets[tid]['_source']['quoted_vids'].split(';')
                video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
                if 'N' in video_ids:
                    video_ids.remove('N')
                text = None
                if retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N':
                    text = tweets[tid]['_source']['retweeted_text']
                elif (retweeted_tweet_id_str == 'N' or retweeted_tweet_id_str == None) and (quoted_tweet_id_str != 'N' and quoted_tweet_id_str != None):
                    #text = tweets[tid]['_source']['quoted_text']
                    text = tweets[tid]['_source']['original_text']
                elif (retweeted_tweet_id_str == 'N' or retweeted_tweet_id_str == None) and (quoted_tweet_id_str == 'N' or quoted_tweet_id_str == None):
                    text = tweets[tid]['_source']['original_text']
                else:
                    text = tweets[tid]['_source']['original_text']

                ## preprocess text with ekphrasis
                preporcessed_text = parser.parseText(text)
                ## get LIWC features
                liwc_measure = liwc.getLIWCFeatures(cat, dic, preporcessed_text, selected_cats=None)
                #liwc_measure['tc'] = 1
                #liwc_measure.update((x, 1) for x, y in liwc_measure.items() if y > 0 and x != 'wc')
                ## get Empath features
                empath_measure = empath_dic.analyze(preporcessed_text)
                #empath_measure['tc'] = 1
                #empath_measure.update((x, 1) for x, y in empath_measure.items() if y > 0)

                '''
                if (retweeted_tweet_id_str == None or retweeted_tweet_id_str == 'N') and (quoted_tweet_id_str != 'N' and quoted_tweet_id_str != None):
                    print('Original:', tweets[tid]['_source']['original_text'])
                    print('Qutoed:', tweets[tid]['_source']['quoted_text'])
                    print('Preprocessed:', preporcessed_text)
                    print('============================================================')
                '''
                '''
                if (retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N'):
                    print('Original:', tweets[tid]['_source']['original_text'])
                    print('Qutoed:', tweets[tid]['_source']['quoted_text'])
                    print('Retweeted:', tweets[tid]['_source']['retweeted_text'])
                    print('Preprocessed:', preporcessed_text)
                    print('https://twitter.com/Deep__AI/status/{}'.format(tid))
                    print('============================================================')
                '''

                for vid in video_ids:
                    if vid in filtered_video_ids:
                        ## find liwc measures
                        if vid not in language_liwc_measures:
                            #language_liwc_measures[vid] = copy.deepcopy(liwc_measure)
                            new_dict = {key: [tid] if liwc_measure[key] > 0 else [] for key in liwc_measure.keys() if key != 'wc' and key != 'dic_wc'}
                            new_dict['wc'] = liwc_measure['wc']
                            new_dict['dic_wc'] = liwc_measure['dic_wc']
                            new_dict['tc'] = 1
                            language_liwc_measures[vid] = new_dict
                        else:
                            #new_dict = {k: language_liwc_measures[vid].get(k, 0) + liwc_measure.get(k, 0) for k in set(liwc_measure)}
                            #language_liwc_measures[vid] = new_dict
                            for key in liwc_measure:
                                if liwc_measure[key] > 0 and key != 'wc' and key != 'dic_wc':
                                    language_liwc_measures[vid][key].append(tid)
                            language_liwc_measures[vid]['wc'] += liwc_measure['wc']
                            language_liwc_measures[vid]['dic_wc'] += liwc_measure['dic_wc']
                            language_liwc_measures[vid]['tc'] += 1
                        
                        ## find empath measures
                        if vid not in language_empath_measures:
                            #language_empath_measures[vid] = copy.deepcopy(empath_measure)
                            new_dict = {key: [tid] if empath_measure[key] > 0 else [] for key in empath_measure.keys()}
                            new_dict['tc'] = 1
                            language_empath_measures[vid] = new_dict
                        else:
                            #new_dict = {k: language_empath_measures[vid].get(k, 0) + empath_measure.get(k, 0) for k in set(empath_measure)}
                            #language_empath_measures[vid] = new_dict
                            for key in empath_measure:
                                if empath_measure[key] > 0:
                                    language_empath_measures[vid][key].append(tid)
                            language_empath_measures[vid]['tc'] += 1

                counter+=1
                print('cnt: {}'.format(counter), end='\r')

            pickle.dump(language_liwc_measures, open(self.ea_language_liwc_measures_path, 'wb'))
            pickle.dump(language_empath_measures, open(self.ea_language_empath_measures_path, 'wb'))
        
        return language_liwc_measures, language_empath_measures
    
    def analyzeLanguageMeasures(self, dict_type, measure_type, video_leanings_probs):
        language_measures = {}
        if dict_type == 'liwc':
            language_measures = pickle.load(open(self.ea_language_liwc_measures_path, 'rb'))
        elif dict_type == 'empath':
            language_measures = pickle.load(open(self.ea_language_empath_measures_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        for vid in language_measures:
            num_tweets_with_measure = None
            if measure_type == 'wc' or measure_type == 'dic_wc':
                num_tweets_with_measure = language_measures[vid][measure_type]
            else:
                num_tweets_with_measure = len(language_measures[vid][measure_type])
            language_measures[vid][measure_type] = float(num_tweets_with_measure) / language_measures[vid]['tc']
        
        measures_left = [language_measures[vid][measure_type] for vid in vids['L']]
        measures_right = [language_measures[vid][measure_type] for vid in vids['R']]
        measures_neutral = [language_measures[vid][measure_type] for vid in vids['N']]
        
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance tests
        print('Significance Test')
        self.util.applySignificanceTest(measures_left, measures_right)
        
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"],'{}'.format(measure_type), None, False)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], '{}'.format(measure_type), None, False)
    
    
    def analyzeTweetsByLanguageCategory(self, dict_type, measure_type, video_leanings_probs, tweets, leaning):
        language_measures = {}
        if dict_type == 'liwc':
            language_measures = pickle.load(open(self.ea_language_liwc_measures_path, 'rb'))
        elif dict_type == 'empath':
            language_measures = pickle.load(open(self.ea_language_empath_measures_path, 'rb'))
        
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        measures_left = [language_measures[vid][measure_type] for vid in vids['L']]
        measures_right = [language_measures[vid][measure_type] for vid in vids['R']]
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))        
        
        all_vids = vids['L'] + vids['R'] + vids['N']
        
        for vid in all_vids:
            print(vid)
            for tid in language_measures[vid][measure_type]:
                retweeted_tweet_id_str = tweets[tid]['_source']['retweeted_tweet_id_str']
                quoted_tweet_id_str = tweets[tid]['_source']['quoted_tweet_id_str']
                reply_user_id_str = tweets[tid]['_source']['reply_user_id_str']
                text = None
                if retweeted_tweet_id_str != None and retweeted_tweet_id_str != 'N':
                    text = tweets[tid]['_source']['retweeted_text']
                elif (retweeted_tweet_id_str == 'N' or retweeted_tweet_id_str == None) and (quoted_tweet_id_str != 'N' and quoted_tweet_id_str != None):
                    #text = tweets[tid]['_source']['quoted_text']
                    text = tweets[tid]['_source']['original_text']
                elif (retweeted_tweet_id_str == 'N' or retweeted_tweet_id_str == None) and (quoted_tweet_id_str == 'N' or quoted_tweet_id_str == None):
                    text = tweets[tid]['_source']['original_text']
                else:
                    text = tweets[tid]['_source']['original_text']
                
                print(text)
                print('-----------------------------------------------------------------------')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    
    ###########################################################################
    ## Video-related operations ###############################################
    ###########################################################################
    ## Analyze Videos by property w.r.t. political leanings (descriptive statistics)
    ## measure_type: [duration, avgWatch, dailyShare, dailyScubscriber, dailyTweet, dailyView, dailyWatch, totalShare,
    ##                totalSubscriber, totalTweet, totalView, commentCount, dislikeCount, favoriteCount, likeCount, viewCount]
    def analyzeVideosByProperty(self, video_leanings_probs, measure_type):
        videos = pickle.load(open(self.videos_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        insights_types = ['avgWatch', 'totalShare', 'totalSubscriber', 'totalTweet', 'totalView', 'totalWatch']
        statistics_types = ['likeCount', 'dislikeCount', 'commentCount', 'favoriteCount', 'viewCount']
        
        video_data = {}
        for vid in video_leanings_probs:
            measure = None
            if measure_type == 'duration':
                measure = self.util.convertDurationToSeconds(videos[vid]['_source']['contentDetails']['duration'])
            elif measure_type in insights_types:
                if measure_type == 'avgWatch':
                    measure = (60 * videos[vid]['_source']['insights']['avgWatch'])
                else:
                    if measure_type == 'totalWatch':
                        watches = [60 * item for item in videos[vid]['_source']['insights']['dailyWatch']]
                        measure = max(1, sum(watches))
                    else:
                        measure = max(1, videos[vid]['_source']['insights'][measure_type])
            elif measure_type in statistics_types:
                if measure_type in videos[vid]['_source']['statistics']:
                    measure = int(videos[vid]['_source']['statistics'][measure_type])
            
            if measure != None:
                video_data[vid] = measure
        
        measures_left = [video_data[vid] for vid in vids['L'] if vid in video_data]
        measures_right = [video_data[vid] for vid in vids['R'] if vid in video_data]
        measures_neutral = [video_data[vid] for vid in vids['N'] if vid in video_data]
        
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance test
        print('Significance Test')
        self.util.applySignificanceTest(measures_left, measures_right)
                
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        ## is_log_chart: boolean
        is_log_chart = False
        if measure_type != 'avgWatch':
            is_log_chart = True
        
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"], measure_type.replace('daily', 'total'), None, is_log_chart)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], measure_type.replace('daily', 'total'), None, is_log_chart)
        
    
    ## Analyze Videos by normalized properties w.r.t. political leanings (descriptive statistics)
    ## measure_type: [avgWatch/duration, totalShare/totalView
    ##                likeCount/viewCount, dislikeCount/viewCount, commentCount/viewCount]
    def analyzeVideosByNormalizedProperty(self, video_leanings_probs, measure_type):
        videos = pickle.load(open(self.videos_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        
        insights_types = ['totalShare']
        statistics_types = ['likeCount', 'dislikeCount', 'commentCount', 'favoriteCount']
        
        video_data = {}
        for vid in video_leanings_probs:
            measure = None
            if measure_type == 'avgWatch':
                awg_watch = (60 * videos[vid]['_source']['insights']['avgWatch'])
                duration = self.util.convertDurationToSeconds(videos[vid]['_source']['contentDetails']['duration'])
                measure = float(awg_watch)/duration
            elif measure_type in insights_types:
                raw_measure = videos[vid]['_source']['insights'][measure_type]
                total_view = videos[vid]['_source']['insights']['totalView']
                measure = float(raw_measure)/total_view
            elif measure_type in statistics_types:
                if measure_type in videos[vid]['_source']['statistics']:
                    raw_measure = float(videos[vid]['_source']['statistics'][measure_type])
                    view_count = int(videos[vid]['_source']['statistics']['viewCount'])
                    measure = raw_measure/view_count
            
            if measure != None:
                video_data[vid] = measure
        
        measures_left = [video_data[vid] for vid in vids['L'] if vid in video_data]
        measures_right = [video_data[vid] for vid in vids['R'] if vid in video_data]
        measures_neutral = [video_data[vid] for vid in vids['N'] if vid in video_data]
        
        print('#measures_left: {}'.format(len(measures_left)))
        print('#measures_right: {}'.format(len(measures_right)))
        print('#measures_neutral: {}'.format(len(measures_neutral)))
        
        ## Significance test
        print('Significance Test')
        self.util.applySignificanceTest(measures_left, measures_right)
        #print(kruskal(measures_left, measures_right, measures_neutral))
                
        ## Draw CDF and KDE for mean_min_cascades, median_min_cascades and max_min_cascades for left vs. right
        self.util.plotCDFMultiple([measures_left, measures_right, measures_neutral], 
                                  ["Left", "Right", "Neutral"], measure_type, None, False)
        self.util.plotHistKDEMulitple([measures_left, measures_right, measures_neutral], 
                                      ["Left", "Right","Neutral"], measure_type, None, False)
        
    
    ## Analyze Videos by normalized properties w.r.t. political leanings (descriptive statistics)
    ## measure_type: [avgWatch/duration, totalShare/totalView
    ##                likeCount/viewCount, dislikeCount/viewCount, commentCount/viewCount]
    def analyzeDivisiveContentsByScatterPlots(self, video_leanings_probs):
        videos = pickle.load(open(self.videos_path, 'rb'))
        vids = self.separateVideosByLeaning(video_leanings_probs)
        tweets = self.getAvailableTweets(0.2)
        video_first_share_time = {}
        for tid in tweets:
            timestamp = int(tweets[tid]['_source']['timestamp_ms'])
            original_video_ids = tweets[tid]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tid]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tid]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            for vid in video_ids:
                if vid not in video_first_share_time:
                    video_first_share_time[vid] = timestamp
                else:
                    if timestamp < video_first_share_time[vid]:
                        video_first_share_time[vid] = timestamp
        
        video_props = {}
        for vid in video_leanings_probs:
            video_props[vid] = {}
            like_count = int(videos[vid]['_source']['statistics']['likeCount'])
            dislike_count = int(videos[vid]['_source']['statistics']['dislikeCount'])
            view_count = int(videos[vid]['_source']['statistics']['viewCount'])
            first_share_time = video_first_share_time[vid]
            polarity, intensity, divisiveness, popularity = self.util.calculateVideoScores(like_count, dislike_count, view_count, first_share_time)
            video_props[vid]['like_count'] = like_count
            video_props[vid]['dislike_count'] = dislike_count
            video_props[vid]['view_count'] = view_count
            video_props[vid]['polarity'] = polarity
            video_props[vid]['intensity'] = intensity
            video_props[vid]['divisiveness'] = divisiveness
            video_props[vid]['popularity'] = popularity
        
        left_vids = []
        right_vids = []
        neutral_vids = []
        
        for vid in vids['L']:
            left_vids.append(vid)
        for vid in vids['R']:
            right_vids.append(vid)
        for vid in vids['N']:
            neutral_vids.append(vid)
        
        all_vids = left_vids + right_vids + neutral_vids
        colors_all = ['blue'] * len(left_vids) + ['red'] * len(right_vids) + ['orange'] * len(neutral_vids)
        
        print('#left_vids: {}, #right_vids: {}, #neutral_vids: {}, #all_vids: {}'.format(len(left_vids), len(right_vids), 
                                                                                        len(neutral_vids), len(all_vids)))
        '''
        ## Draw Scatter plots of #likes and #dislikes for left vs. right
        self.util.plotScatterWithMarkerSize([video_props[vid]['like_count'] for vid in left_vids], 
                                            [video_props[vid]['dislike_count'] for vid in left_vids], 
                                            self.util.scaleMinMax([video_props[vid]['view_count'] for vid in left_vids],1,200), 
                                            "#likes", "#dislikes", None, False, 'blue')
        self.util.plotScatterWithMarkerSize([video_props[vid]['like_count'] for vid in right_vids], 
                                            [video_props[vid]['dislike_count'] for vid in right_vids], 
                                            self.util.scaleMinMax([video_props[vid]['view_count'] for vid in right_vids],1,200), 
                                            "#likes", "#dislikes", None, False, 'red')
        self.util.plotScatterWithMarkerSize([video_props[vid]['like_count'] for vid in neutral_vids],
                                            [video_props[vid]['dislike_count'] for vid in neutral_vids],
                                            self.util.scaleMinMax([video_props[vid]['view_count'] for vid in neutral_vids],1,200), 
                                            "#likes", "#dislikes", None, False, 'orange')
        self.util.plotScatterWithMarkerSize([video_props[vid]['like_count'] for vid in all_vids],
                                            [video_props[vid]['dislike_count'] for vid in all_vids], 
                                            self.util.scaleMinMax([video_props[vid]['view_count'] for vid in all_vids],1,200), 
                                            "#likes", "#dislikes", None, False, colors_all)
        '''
        polarity_left = [video_props[vid]['polarity'] for vid in left_vids]
        polarity_right = [video_props[vid]['polarity'] for vid in right_vids]
        polarity_neutral = [video_props[vid]['polarity'] for vid in neutral_vids]
        polarity_all = [video_props[vid]['polarity'] for vid in all_vids]
        intensity_left = [video_props[vid]['intensity'] for vid in left_vids]
        intensity_right = [video_props[vid]['intensity'] for vid in right_vids]
        intensity_neutral = [video_props[vid]['intensity'] for vid in neutral_vids]
        intensity_all = [video_props[vid]['intensity'] for vid in all_vids]
        divisiveness_left = [video_props[vid]['divisiveness'] for vid in left_vids]
        divisiveness_right = [video_props[vid]['divisiveness'] for vid in right_vids]
        divisiveness_neutral = [video_props[vid]['divisiveness'] for vid in neutral_vids]
        divisiveness_all = [video_props[vid]['divisiveness'] for vid in all_vids]
        popularity_left = [video_props[vid]['popularity'] for vid in left_vids]
        popularity_right = [video_props[vid]['popularity'] for vid in right_vids]
        popularity_neutral = [video_props[vid]['popularity'] for vid in neutral_vids]
        popularity_all = [video_props[vid]['popularity'] for vid in all_vids]
        
        ## Significance test
        print('Significance Polarity')
        self.util.applySignificanceTest(polarity_left, polarity_right)
        print('Significance Intensity')
        self.util.applySignificanceTest(intensity_left, intensity_left)
        print('Significance divisiveness')
        self.util.applySignificanceTest(divisiveness_left, divisiveness_right)
        print('Significance popularity')
        self.util.applySignificanceTest(popularity_left, popularity_right)
        
        ## Correlation
        dataset_left = pd.DataFrame({'Polarity': polarity_left, 'Intensity': intensity_left, 
                                     'Divisiveness': divisiveness_left, 'Popularity': popularity_left})
        dataset_right = pd.DataFrame({'Polarity': polarity_right, 'Intensity': intensity_right, 
                                     'Divisiveness': divisiveness_right, 'Popularity': popularity_right})
        print('Correlation Left')
        self.util.calculateCorrelation(dataset_left, method='spearman')
        print('Correlation Right')
        self.util.calculateCorrelation(dataset_right, method='spearman')
        
        '''
        ## Partial Correlation
        dataset_left = pd.DataFrame({'Polarity': polarity_left, 'Intensity': intensity_left, 
                                     'Divisiveness': divisiveness_left, 'Popularity': popularity_left})
        dataset_right = pd.DataFrame({'Polarity': polarity_right, 'Intensity': intensity_right, 
                                     'Divisiveness': divisiveness_right, 'Popularity': popularity_right})
        dataset_all = pd.DataFrame({'Polarity': polarity_all, 'Intensity': intensity_all, 
                                    'Divisiveness': divisiveness_all, 'Popularity': popularity_all})
        print('Partial Correlation Left')
        self.util.partial_corr(dataset_left, method='spearman')
        print('Partial Correlation Right')
        self.util.partial_corr(dataset_right, method='spearman')
        print('Partial Correlation All')
        self.util.partial_corr(dataset_all, method='spearman')
        '''
        
        ## Draw Intensity-Polarity-Divisiveness
        self.util.plotScatter(polarity_left, intensity_left, None, "Polarity", "Intensity", None, False,
                              [score+0.5 for score in divisiveness_left])
        
        self.util.plotScatter(polarity_right, intensity_right, None, "Polarity", "Intensity", None, False,
                              [score+0.5 for score in divisiveness_right])
        
        self.util.plotScatter(polarity_neutral, intensity_neutral, None, "Polarity", "Intensity", None, False, 
                              [score+0.5 for score in divisiveness_neutral])
        
        self.util.plotScatter(polarity_all, intensity_all, None, "Polarity", "Intensity", None, False,
                              [score+0.5 for score in divisiveness_all])
        
        '''
        ## Draw Popularity-Polarity-Divisiveness
        self.util.plotScatter(polarity_left, popularity_left, None, "Polarity (P)", "Popularity", None, False,
                              [score+0.5 for score in divisiveness_left])
        
        self.util.plotScatter(polarity_right, popularity_right, None, "Polarity (P)", "Popularity", None, False,
                              [score+0.5 for score in divisiveness_right])
        
        self.util.plotScatter(polarity_neutral, popularity_neutral, None, "Polarity (P)", "Popularity", None, False, 
                              [score+0.5 for score in divisiveness_neutral])
        
        self.util.plotScatter(polarity_all, popularity_all, None, "Polarity (P)", "Popularity", None, False,
                              [score+0.5 for score in divisiveness_all])
        
        '''
        
        '''
        ## Print Divisive Top-K videos for each leaning.
        sorted_divisiveness_left = sorted(divisiveness_left.items(), key=itemgetter(1), reverse=True)
        sorted_divisiveness_right = sorted(divisiveness_right.items(), key=itemgetter(1), reverse=True)
        sorted_divisiveness_neutral = sorted(divisiveness_neutral.items(), key=itemgetter(1), reverse=True)
        sorted_divisiveness_all = sorted(divisiveness_all.items(), key=itemgetter(1), reverse=True)
        print('--- Divisive Contents ---')
        top_k = 5
        print('--- Left ---')
        for i in range(top_k):
            vid = sorted_divisive_left[i][0]
            print(vid, likeCounts[vid], dislikeCounts[vid], viewCounts[vid], sorted_divisive_left[i][1], video_leanings_probs[vid]['right'])
        print('--- Right ---')
        for i in range(top_k):
            vid = sorted_divisive_right[i][0]
            print(vid, likeCounts[vid], dislikeCounts[vid], viewCounts[vid], sorted_divisive_right[i][1], video_leanings_probs[vid]['right'])
        print('--- Neutral ---')
        for i in range(top_k):
            vid = sorted_divisive_neutral[i][0]
            print(vid, likeCounts[vid], dislikeCounts[vid], viewCounts[vid], sorted_divisive_neutral[i][1], video_leanings_probs[vid]['right'])
        print('--- All ---')
        for i in range(top_k):
            vid = sorted_divisive_all[i][0]
            print(vid, likeCounts[vid], dislikeCounts[vid], viewCounts[vid], sorted_divisive_all[i][1], video_leanings_probs[vid]['right'])
        '''
    
    
    ###########################################################################
    ## FOLLOWERS/FRIENDS Graph and Communities related operations #############
    ###########################################################################
    ## Summarize communities(clusters) w.r.t hashtags in user profile descriptions, tweets, retweets, verified users, 
    ## followers, etc.
    def summarizeCommunities(self):
        users = pickle.load(open(self.users_path, 'rb'))
        users_locs = pickle.load(open(self.ea_users_locs_path, 'rb'))
        tweets = pickle.load(open(self.ea_tweets_path, 'rb'))
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        
        connection_list = pickle.load(open(self.ea_users_followers_path, 'rb'))
        
        communities = list(set(list(user_comm_pairs['assigned_com_memberships'].values())))
        num_communities = len(communities)
        hashtags = {}
        users_verified = {}
        users_with_hashtag = {}
        tweet_counts = {}
        retweets_per_tweet = {}
        original_tweets = {}
        com_users = {}
        followers = {}
        locations = {}
        tweets_per_user = {}
        followers_of_verified_users = {}
        for com in communities:
            hashtags[com] = {}
            users_with_hashtag[com] = 0
            users_verified[com] = 0
            tweet_counts[com] = 0
            retweets_per_tweet[com] = 0
            original_tweets[com] = 0
            com_users[com] = 0
            followers[com] = []
            locations[com] = {}
            tweets_per_user[com] = {}
            followers_of_verified_users[com] = []
        for user_id in user_comm_pairs['assigned_com_memberships']:
            ## hashtags
            user_com = user_comm_pairs['assigned_com_memberships'][user_id]
            user_desc = users[user_id]['_source']['description']
            user_hashtags = list(set(re.findall(r"#(\w+)", user_desc)))
            user_hashtags = [tag.lower() for tag in user_hashtags]
            for tag in user_hashtags:
                if tag not in hashtags[user_com]:
                    hashtags[user_com][tag] = 1
                else:
                    hashtags[user_com][tag] += 1
            ## number of users with hashtag in their description
            if len(user_hashtags) > 0:
                users_with_hashtag[user_com] += 1
                        
            ## number of users with verified accounts
            user_ver = users[user_id]['_source']['verified'].strip()
            if user_ver != 'N':
                users_verified[user_com] += 1
                followers_of_verified_users[user_com].append(len(connection_list[user_id]))
            
            ## number of followers
            num_connections = len(connection_list[user_id])
            #num_connections = int(users[user_id]['_source']['followers_count'])
            followers[user_com].append(num_connections)
            
            
            ## users locations from user profiles
            user_loc = users_locs[user_id]
            if user_loc != None and user_loc != "United States" and user_loc != "USA":
                if user_loc in locations[user_comm_pairs['assigned_com_memberships'][user_id]]:
                    locations[user_com][user_loc] += 1
                else:
                    locations[user_com][user_loc] = 1
            
            com_users[user_com] += 1
        
        ## number of tweets in each cluster and number of retweets per tweet
        print(len(user_comm_pairs['assigned_com_memberships'].keys()))
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            #if user_id in user_comm_pairs['assigned_com_memberships']:
            user_com = user_comm_pairs['assigned_com_memberships'][user_id]
            tweet_counts[user_com] += 1

            if user_id not in tweets_per_user[user_com]:
                tweets_per_user[user_com][user_id] = 1
            else:
                tweets_per_user[user_com][user_id] += 1
            
            retweeted_user_id = tweets[tweet_id]['_source']['retweeted_user_id_str'].strip()
            if retweeted_user_id != 'N' and retweeted_user_id in user_comm_pairs['assigned_com_memberships']:
                com = user_comm_pairs['assigned_com_memberships'][retweeted_user_id]
                retweets_per_tweet[com] += 1
            elif retweeted_user_id == 'N' and user_id in user_comm_pairs['assigned_com_memberships']:
                com = user_comm_pairs['assigned_com_memberships'][user_id]
                original_tweets[com] += 1
        
        for com in hashtags:
            print("Community:", com)
            print("# of users:", com_users[com])
            print("# of tweets:", tweet_counts[com])
            print("# of verified users:", users_verified[com])
            print("mean of followers of verified users: {}".format(np.mean(np.array(followers_of_verified_users[com]))), "median of followers of verified users: {}".format(np.median(np.array(followers_of_verified_users[com]))))
            print("# of users with hashtags in their profiles:", users_with_hashtag[com])
            print("# of retweets:", retweets_per_tweet[com])
            print("# of original tweets:", original_tweets[com])
            print("# of retweets per tweet:", float(retweets_per_tweet[com]) / original_tweets[com])
            print("mean of followers: {}".format(np.mean(np.array(followers[com]))), "median of followers: {}".format(np.median(np.array(followers[com]))))
            sorted_locations = sorted(locations[com].items(), key=itemgetter(1), reverse=True)
            print(sum(list(locations[com].values())), sorted_locations[:25])
            sorted_hashtags = sorted(hashtags[com].items(), key=itemgetter(1), reverse=True)
            print(sorted_hashtags)
            print('---------------------------------------------------------------------')
            
        return followers, tweets_per_user
    
    
    ## Return #locations of users for each state and for each community.
    ## measure_type = [user | tweet | retweet]
    def getLocationsByCommunities(self, tweets, measure_type):
        users = pickle.load(open(self.users_path, 'rb'))
        users_locs = pickle.load(open(self.ea_users_locs_path, 'rb'))
        #tweets = pickle.load(open(self.tweets_path, 'rb'))
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        
        communities = list(set(list(user_comm_pairs['assigned_com_memberships'].values())))
        num_communities = len(communities)
        locations = {}
        for com in communities:
            locations[com] = {}
        
        if measure_type == 'user':
            for user_id in user_comm_pairs['assigned_com_memberships']:
                user_loc = users_locs[user_id]
                user_com = user_comm_pairs['assigned_com_memberships'][user_id]
                if user_loc != None:
                    if user_loc in locations[user_com]:
                            locations[user_com][user_loc] += 1
                    else:
                        locations[user_com][user_loc] = 1
        else:
            for tweet_id in tweets:
                user_id = tweets[tweet_id]['_source']['user_id_str']
                user_loc = users_locs[user_id]
                if user_loc != None:
                    #if user_id in user_comm_pairs['assigned_com_memberships']:
                    user_com = user_comm_pairs['assigned_com_memberships'][user_id]
                    if measure_type == 'retweet':
                        if tweets[tweet_id]['_source']['retweeted_text'].strip() != 'N':
                            if user_loc in locations[user_com]:
                                locations[user_com][user_loc] += 1
                            else:
                                locations[user_com][user_loc] = 1
                    elif measure_type == 'tweet':
                        if user_loc in locations[user_com]:
                            locations[user_com][user_loc] += 1
                        else:
                            locations[user_com][user_loc] = 1
        
        return locations
    
    
    ## Return the most shared videos for each community.
    def getMostSharedVideosByCommunities(self, tweets):
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        
        communities = list(set(list(user_comm_pairs['assigned_com_memberships'].values())))
        num_communities = len(communities)
        videos = {}
        for com in communities:
            videos[com] = {}
        
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            #if user_id in user_comm_pairs['assigned_com_memberships']:
            user_com = user_comm_pairs['assigned_com_memberships'][user_id]
            original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            for video_id in video_ids:
                if video_id != 'N':
                    if video_id in videos[user_com]:
                        videos[user_com][video_id] += 1
                    else:
                        videos[user_com][video_id] = 1
        
        return videos
    
    
    ## check #users from each political leaning in the communities.
    def checkLeaningsInCommunities(self):
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        users_leanings_labels = pickle.load(open(self.ea_seed_users_leanings_labels_path, 'rb'))
        users_inferred_leanings_scores = pickle.load(open(self.ea_users_inferred_leanings_scores_path, 'rb'))
        
        communities = list(set(list(user_comm_pairs['assigned_com_memberships'].values())))
        num_communities = len(communities)
        predefined_leanings = {}
        inferred_leanings = {}
        for com in communities:
            predefined_leanings[com] = {'left': 0, 'right': 0}
            inferred_leanings[com] = {'left': 0, 'right': 0}
        
        for user_id in users_inferred_leanings_scores:
            user_com = user_comm_pairs['assigned_com_memberships'][user_id]
            
            first_leaning = users_leanings_labels[user_id]
            if first_leaning == 0:
                predefined_leanings[user_com]['left'] += 1
            elif first_leaning == 1:
                predefined_leanings[user_com]['right'] += 1
            
            leaning_scores = users_inferred_leanings_scores[user_id]
            if leaning_scores['left'] > leaning_scores['right']:
                inferred_leanings[user_com]['left'] += 1
            elif leaning_scores['left'] < leaning_scores['right']:
                inferred_leanings[user_com]['right'] += 1
        
        for com in inferred_leanings:
            print("Community:", com)
            print("Predefined Leanings -- left: {}".format(predefined_leanings[com]['left']), "right: {}".format(predefined_leanings[com]['right']), 'total: {}'.format(predefined_leanings[com]['left'] + predefined_leanings[com]['right']))    
            print("Inferred Leanings -- left: {}".format(inferred_leanings[com]['left']), "right: {}".format(inferred_leanings[com]['right']), 'total: {}'.format(inferred_leanings[com]['left'] + inferred_leanings[com]['right'])) 
        
        return predefined_leanings, inferred_leanings
    
    
    ## Return the tweets posted by a super-community for a given video-id.
    def getTweetsOfCommunitiesByVideoId(self, tweets, video_id, super_com):
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        users = pickle.load(open(self.users_path, 'rb'))
        
        results = []
        
        for tweet_id in tweets:
            user_id = tweets[tweet_id]['_source']['user_id_str']
            user_screen_name = users[user_id]['_source']['screen_name']
            user_com = user_comm_pairs['assigned_com_memberships'][user_id]
            original_video_ids = tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            if video_id in video_ids and user_com in super_com:
                '''
                print(tweet_id)
                print("https://twitter.com/{}/status/{}".format(user_screen_name, tweet_id))
                print(tweets[tweet_id]['_source']['original_text'])
                if tweets[tweet_id]['_source']['retweeted_text'].strip() != 'N':
                    print(tweets[tweet_id]['_source']['retweeted_text'])
                if tweets[tweet_id]['_source']['quoted_text'].strip() != 'N':
                    print(tweets[tweet_id]['_source']['quoted_text'])
                print('--------------------------------------------------------')
                '''
                #results.append(tweets[tweet_id]['_source']['original_text'])
                results.append(tweets[tweet_id])
        
        return results
    
    
    ## Return the tweet cascade of a super-community for a given video-id.
    def getTweetCascadesOfCommunitiesByVideoId(self, tweets, video_id, super_com):
        related_tweets = self.getTweetsOfCommunitiesByVideoId(tweets, video_id, super_com)
        
        connection_list = pickle.load(open(self.ea_users_followers_path, 'rb'))
        
        user_time_dict = {}
        for tweet in related_tweets:
            user_id = tweet['_source']['user_id_str']
            if user_id not in user_time_dict:
                user_time_dict[user_id] = int(tweet['_source']['timestamp_ms'])
            else:
                if int(tweet['_source']['timestamp_ms']) < user_time_dict[user_id]:
                    user_time_dict[user_id] = int(tweet['_source']['timestamp_ms'])
        
        print("Users sorted by timestamp_ms! #of users: {}".format(len(user_time_dict.keys())))
        
        user_ids = [item[0] for item in sorted(user_time_dict.items(), key=itemgetter(1))]
        user_follower_mat = np.zeros(shape=(len(user_ids), len(user_ids)))
        
        ## user_follower_mat[i][j] ==> user i follows user j
        for i in range(len(user_ids)):
            if i % 100 == 0:
                print("Follower vector created for user {}!".format(str(i)))
            for j in range(len(user_ids)):
                if i!=j and user_ids[i] in connection_list[user_ids[j]]:
                        user_follower_mat[i][j] = 1.
        
        print("Follower matrix created!")
        
        print(np.amin(np.sum(user_follower_mat, axis=0)), np.amax(np.sum(user_follower_mat, axis=0)), np.mean(np.sum(user_follower_mat, axis=0)), np.median(np.sum(user_follower_mat, axis=0)))
        
        all_cascade_list = {}
        for i in range(len(user_ids)):
            all_cascade_list[i] = {}
        
        for i in range(len(user_ids)):
            if i % 10 == 0:
                print("User {} is processed!".format(str(i)))
            followers = np.where(user_follower_mat[i]==1)[0]
            followers = followers[followers<i]
            min_cascades = []
            max_cascades = []
            for fol_ind in followers:
                min_cascades.append(all_cascade_list[fol_ind]['min'])
                max_cascades.append(all_cascade_list[fol_ind]['max'])
            
            if len(followers) == 0:
                all_cascade_list[i]['min'] = 1
                all_cascade_list[i]['max'] = 1
            else:
                all_cascade_list[i]['min'] = min(min_cascades) + 1
                all_cascade_list[i]['max'] = max(max_cascades) + 1
            
        return user_ids, all_cascade_list
    
    
    ###########################################################################
    ## ADDITIONAL operations ##################################################
    ###########################################################################
    ## This checks the change in follower/friends w.r.t time since the user profiles and followers 
    ## are collected in different times
    def checkConnectionChanges(self):
        users = pickle.load(open(self.users_path, 'rb'))
        connection_list = pickle.load(open(self.ea_users_followers_path, 'rb'))
        user_comm_pairs = pickle.load(open(self.ea_communities_path, 'rb'))
        communities = list(set(list(user_comm_pairs['assigned_com_memberships'].values())))
        
        total_num_diffs = []
        total_perc_diffs = []
        comm_num_diffs = {}
        comm_perc_diffs = {}
        for com in communities:
            comm_num_diffs[com] = []
            comm_perc_diffs[com] = []
        
        for user_id in connection_list:
            cur_num_connections = len(connection_list[user_id])
            past_num_connections = users[user_id]['_source']['followers_count'] if connection_type == 'followers' else users[user_id]['_source']['friends_count']
            
            try:
                past_num_connections = int(past_num_connections)
                num_diff = abs(cur_num_connections - past_num_connections)
                base = cur_num_connections if cur_num_connections < past_num_connections else past_num_connections
                perc_diff = None
                if base != 0:
                    perc_diff = (float(num_diff) / base) * 100
                else:
                    perc_diff = float(num_diff) * 100
                total_num_diffs.append(num_diff)
                total_perc_diffs.append(perc_diff)
                if user_id in user_comm_pairs['assigned_com_memberships']:
                    com = user_comm_pairs['assigned_com_memberships'][user_id]
                    comm_num_diffs[com].append(num_diff)
                    comm_perc_diffs[com].append(perc_diff)
            except Exception as e:
                print(user_id, e)
        
        total_num_diffs = np.array(total_num_diffs)
        total_perc_diffs = np.array(total_perc_diffs)
        
        print('------ Total available user statistics --------')
        print('mean(num_diffs):', np.mean(total_num_diffs))
        print('median(num_diffs):', np.median(total_num_diffs))
        #print('std(num_diffs):', np.std(num_diffs))
        print('mean(perc_diffs):', np.mean(total_perc_diffs))
        print('median(perc_diffs):', np.median(total_perc_diffs))
        #print('std(perc_diffs):', np.std(perc_diffs))
        
        for com in comm_num_diffs:  
            print('------ Community ' + str(com) +  ' statistics --------')
            print('mean(num_diffs):', np.mean(np.array(comm_num_diffs[com])))
            print('median(num_diffs):', np.median(np.array(comm_num_diffs[com])))
            print('mean(perc_diffs):', np.mean(np.array(comm_perc_diffs[com])))
            print('median(perc_diffs):', np.median(np.array(comm_perc_diffs[com])))
    
    
        
