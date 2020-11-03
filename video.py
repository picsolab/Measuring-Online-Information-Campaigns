import json
import pickle
import numpy as np
import math
import random
random.seed(777)
import csv
import datetime
import copy
import dateutil.parser
from operator import itemgetter
import os

class Video:
    def __init__(self, util):
        self.util = util
        
        # Initial videos by keywords and all relevant videos.
        self.initial_videos_path = 'data/social_media/{}/videos_by_keywords.pkl'.format(self.util.campaign)
        self.videos_path = 'data/social_media/{}/all_videos_by_relevant_videos.pkl'.format(self.util.campaign)
        
        # From Siqi
        self.view_120_path = 'data/social_media/{}/videos_view120.pkl'.format(self.util.campaign)
        self.virality_path = 'data/social_media/{}/videos_label_exo_endo.csv'.format(self.util.campaign)
        
        self.bin_size = self.util.bin_size
        self.daily_dates = self.util.daily_dates
        self.aggregated_dates = self.util.getAggregatedDates()
    
    def getVideoVolumeDistribution(self, data):

        daily_counts = np.zeros((len(self.daily_dates)))
        for item in data.values():
            #print(item)
            created_at = item['_source']['snippet']['publishedAt']
            created_at = dateutil.parser.parse(created_at).strftime('%Y-%m-%d')
            video_id = item['_source']['id']
            daily_counts[self.daily_dates.index(created_at)] += 1
        
        print('# items:', len(data.keys()))
        #print('# leap_items:', daily_counts[-1])
        
        ## weekly aggregation
        aggregated_counts = self.util.getAggregatedCounts(daily_counts)
        print('# of total videos:', np.sum(aggregated_counts))
        
        return aggregated_counts
    
    ## get total video watches
    def getTotalVideoWatches(self, videos):
        count = 0
        video_watch_mat = np.zeros(shape=(len(videos.keys()), len(self.daily_dates)))
        for video_id in videos:
            watch_vec = np.zeros(shape=(len(self.daily_dates)))
            start_date = str(videos[video_id]['_source']['insights']['startDate'])
            start_idx = self.daily_dates.index(start_date)
            daily_watches = videos[video_id]['_source']['insights']['dailyWatch']
            days = videos[video_id]['_source']['insights']['days']
            #print(start_idx + len(days), len(self.daily_dates))
            if (start_idx + len(days)) <= len(self.daily_dates):
                watch_vec[start_idx:(start_idx + len(days))] = np.array(daily_watches)
            else:
                watch_vec[start_idx:] = np.array(daily_watches[:(len(self.daily_dates)-start_idx)])
            video_watch_mat[count] = watch_vec
            count+=1
        
        aggregated_watches = self.util.getAggregatedCounts(np.sum(video_watch_mat, axis=0))
        print('# of total video watches:', np.sum(aggregated_watches))
        #aggregated_watches[aggregated_watches==0] = 1
        #return np.log10(aggregated_watches)
        return aggregated_watches
    
    ## get total video views
    def getTotalVideoViews(self, videos):
        count = 0
        video_view_mat = np.zeros(shape=(len(videos.keys()), len(self.daily_dates)))
        for video_id in videos:
            view_vec = np.zeros(shape=(len(self.daily_dates)))
            start_date = str(videos[video_id]['_source']['insights']['startDate'])
            start_idx = self.daily_dates.index(start_date)
            daily_views = videos[video_id]['_source']['insights']['dailyView']
            days = videos[video_id]['_source']['insights']['days']
            #print(start_idx + len(days), len(self.daily_dates))
            if (start_idx + len(days)) <= len(self.daily_dates):
                view_vec[start_idx:(start_idx + len(days))] = np.array(daily_views)
            else:
                view_vec[start_idx:] = np.array(daily_views[:(len(self.daily_dates)-start_idx)])
            video_view_mat[count] = view_vec
            count+=1
            
        aggregated_views = self.util.getAggregatedCounts(np.sum(video_view_mat, axis=0))
        print('# of total video views:', np.sum(aggregated_views))
        #aggregated_views[aggregated_views==0] = 1
        #return np.log10(aggregated_views)
        return aggregated_views
    
    ## get total video shares
    def getTotalVideoShares(self, videos):
        count = 0
        video_share_mat = np.zeros(shape=(len(videos.keys()), len(self.daily_dates)))
        for video_id in videos:
            view_vec = np.zeros(shape=(len(self.daily_dates)))
            start_date = str(videos[video_id]['_source']['insights']['startDate'])
            start_idx = self.daily_dates.index(start_date)
            daily_shares = videos[video_id]['_source']['insights']['dailyShare']
            days = videos[video_id]['_source']['insights']['days']
            #print(start_idx + len(days), len(self.daily_dates))
            if (start_idx + len(days)) <= len(self.daily_dates):
                view_vec[start_idx:(start_idx + len(days))] = np.array(daily_shares)
            else:
                view_vec[start_idx:] = np.array(daily_shares[:(len(self.daily_dates)-start_idx)])
            video_share_mat[count] = view_vec
            count+=1
        
        aggregated_shares = self.util.getAggregatedCounts(np.sum(video_share_mat, axis=0))
        print('# of total video shares:', np.sum(aggregated_shares))
        #aggregated_shares[aggregated_shares==0] = 1
        #return np.log10(aggregated_shares)
        return aggregated_shares
