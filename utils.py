import json
import pickle
import numpy as np
import pandas as pd
import math
import random
random.seed(777)
import csv
import datetime
import copy
import dateutil.parser
from operator import itemgetter
import os
import re
import matplotlib.pyplot as plt
from matplotlib import mlab
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu, kruskal, ks_2samp, spearmanr, pearsonr
from scipy.stats import entropy
import pingouin as pg

class Utils:
    def __init__(self, campaign, bin_size, connection_type, year):
        self.campaign = campaign
        self.bin_size = bin_size
        self.connection_type = connection_type
        self.year = year
        self.daily_dates = [str(date) for date in np.arange('2017-01-01', '2018-05-01', dtype='datetime64')]
        self.aggregated_dates = self.getAggregatedDates()
    
    def getAggregatedDates(self):
        aggregated_dates = []
        for i in range(0, len(self.daily_dates), self.bin_size):
            if (i+self.bin_size) < len(self.daily_dates):
                date = self.daily_dates[i] + ' - ' + self.daily_dates[i + self.bin_size -1]
            else:
                date = self.daily_dates[i] + ' - ' + self.daily_dates[-1]
            aggregated_dates.append(date)
        return aggregated_dates
    
    def getAggregatedCounts(self, daily_counts):
        ## weekly aggregation
        aggregated_counts = np.zeros((math.ceil(len(self.daily_dates) / self.bin_size)))
        for i in range(0, len(aggregated_counts)):
            if (i+1)*self.bin_size <= len(self.daily_dates):
                aggregated_counts[i] = np.sum(daily_counts[i*self.bin_size:(i+1)*self.bin_size])
            else:
                aggregated_counts[i] = np.sum(daily_counts[i*self.bin_size:])

        #print('# of total events:', np.sum(aggregated_counts))
        return aggregated_counts
    
    ## event_type = ['mass_shooting_count' | 'mass_shooting_volume' | 'total_gv_count' | 'total_gv_volume'] for gun
    def getOfflineEventsVolumeDistribution(self, event_type):

        daily_counts = np.zeros((len(self.daily_dates)))

        if self.campaign == 'blm':
            blm_protests = json.load(open('data/elephrame/blm.json', 'r'))
            for protest in blm_protests:
                start_date_raw = protest['start_date'].split('/')
                start_date = start_date_raw[2] + '-' + start_date_raw[1] + '-' + start_date_raw[0]
                if start_date in self.daily_dates:
                    daily_counts[self.daily_dates.index(start_date)] += 1

        elif self.campaign == 'gun':
            valid_years = ['2017', '2018']
            if event_type == 'mass_shooting_count' or event_type == 'mass_shooting_volume':
                for year in valid_years:
                    with open('data/gva/mass_shootings_' + year + '.csv') as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        next(csv_reader)
                        line_count = 0
                        for row in csv_reader:
                            start_date = row[0].replace(',', '').split(' ')
                            start_date =  start_date[0][:3] + ' ' + start_date[1] + ' ' + start_date[2]
                            start_date = datetime.datetime.strptime(start_date, '%b %d %Y').strftime('%Y-%m-%d')
                            if start_date in self.daily_dates:
                                if event_type == 'mass_shooting_count':
                                    daily_counts[self.daily_dates.index(start_date)] += 1
                                elif event_type == 'mass_shooting_volume':
                                    daily_counts[self.daily_dates.index(start_date)] += (int(row[4]) + int(row[5]))
                                line_count += 1
                    #print("# of mass shootings:", line_count)
            elif event_type == 'total_gv_count' or event_type == 'total_gv_volume':
                for year in valid_years:
                    with open('data/gva/all_shootings_' + year + '.csv') as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        next(csv_reader)
                        line_count = 0
                        for row in csv_reader:
                            start_date = row[1].replace(',', '').split(' ')
                            start_date =  start_date[0][:3] + ' ' + start_date[1] + ' ' + start_date[2]
                            start_date = datetime.datetime.strptime(start_date, '%b %d %Y').strftime('%Y-%m-%d')
                            if start_date in self.daily_dates:
                                if event_type == 'total_gv_count':
                                    daily_counts[self.daily_dates.index(start_date)] += 1
                                elif event_type == 'total_gv_volume':
                                    daily_counts[self.daily_dates.index(start_date)] += (int(row[5]) + int(row[6]))
                                line_count += 1
                    #print("# of mass shootings:", line_count)

        elif self.campaign == 'img':
            trump_protests = json.load(open('data/elephrame/trump_admin.json', 'r'))

        ## weekly aggregation
        aggregated_counts = self.getAggregatedCounts(daily_counts)
        return aggregated_counts
    
    ## Sort activity_counts (tweets, videos, offline events) and weekly dates, return both
    def sortOfflineEventVolumeByDate(self, activity_counts):
        #print(self.aggregated_dates)
        #print(activity_counts)
        sorted_activities_counts_ind = np.argsort(activity_counts)[::-1]
        weeks_sorted = itemgetter(*sorted_activities_counts_ind)(self.aggregated_dates)
        for i in sorted_activities_counts_ind:
            print(i, self.aggregated_dates[i], activity_counts[i])
        
        return sorted_activities_counts_ind, weeks_sorted
    
    ## Sort activity_counts (tweets, videos, offline events) and weekly dates, return both
    def sortVolumeByDate(self, activity_counts):
        #print(self.aggregated_dates)
        #print(activity_counts)
        sorted_activities_counts_ind = np.argsort(activity_counts)[::-1]
        weeks_sorted = itemgetter(*sorted_activities_counts_ind)(self.aggregated_dates)
        for i in sorted_activities_counts_ind:
            print(i, self.aggregated_dates[i], activity_counts[i])
        
        return sorted_activities_counts_ind, weeks_sorted
    
    
    ## Merge communities irrespective of the purpose to obtain measuers for supercommunities
    def mergeMeasuresForSuperCommunities(self, com_measures, super_communities):
        result = []
        len_super_coms = len(super_communities)
        for i in range(len_super_coms):
            super_com_result = {}
            for com in super_communities[i]:
                for item in com_measures[com]:
                    if item not in super_com_result:
                        super_com_result[item] = com_measures[com][item]
                    else:
                        super_com_result[item] += com_measures[com][item]
            
            sorted_super_com_result = sorted(super_com_result.items(), key=itemgetter(1), reverse=True)
            result.append(sorted_super_com_result)
        
        return result
    
    ## calculate gini index of a given list
    def gini(self, list_of_values):
        #sorted_list = sorted(list_of_values)
        sorted_list = np.sort(list_of_values)
        if np.sum(sorted_list) == 0.:
            return 0.
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
        fair_area = height * len(list_of_values) / 2.
        return (fair_area - area) / fair_area
    
    ## calculate Polarity, Intensity, Divisiveness and Popularity Scores
    def calculateVideoScores(self, like_count, dislike_count, view_count, first_share_time):
        polarity = entropy([like_count, dislike_count], base=2)
        #intensity = float(like_count + dislike_count) / np.log2(view_count)
        intensity = float(like_count + dislike_count) / view_count
        divisiveness = None
        if like_count + dislike_count != 0:
            divisiveness = (float(dislike_count) / (like_count + dislike_count)) - 0.5
            #divisiveness = np.log2((float(dislike_count)/(like_count + dislike_count)) / (float(like_count)/(like_count + dislike_count)))
        popularity = float(view_count) / (float((datetime.datetime(2018, 5, 1, 0, 0, 0, 0).timestamp() * 1000) - first_share_time) / (24 * 60 * 60 * 1000))
        return polarity, intensity, divisiveness, popularity
    
    def bootstrap(self, data1, data2, n=1000, func=np.mean):
        """
        Generate 'n' bootstrap samples, evaluating 'func'
        at each resampling. 'bootstrap' returns a function,
        which can be called to obtain confidence intervals
        of interest.
        """
        simulations = list()
        sample_size1 = len(data1)
        sample_size2 = len(data2)
        #xbar_init_1 = np.mean(data_1)
        #xbar_init_2 = np.mean(data_2)
        for c in range(n):
            itersample1 = np.random.choice(data1, size=sample_size1, replace=True)
            itersample2 = np.random.choice(data2, size=sample_size2, replace=True)
            simulations.append(func(itersample1)-func(itersample2))
        simulations.sort()
        def ci(p):
            """
            Return 2-sided symmetric confidence interval specified
            by p.
            """
            u_pval = (1+p)/2.
            l_pval = (1-u_pval)
            l_indx = int(np.floor(n*l_pval))
            u_indx = int(np.floor(n*u_pval))
            return(simulations[l_indx],simulations[u_indx])
        return(ci)
    
    def applySignificanceTest(self, data1, data2):
        print('Less:', mannwhitneyu(data1, data2, alternative='less'))
        print('Greater:', mannwhitneyu(data1, data2, alternative='greater'))
        print('Two-sided:', mannwhitneyu(data1, data2, alternative='two-sided'))
        print(ks_2samp(data1, data2,))
        boot_median = self.bootstrap(data1, data2, n=5000, func=np.median)
        cinterval_median = boot_median(.95)
        print("Quantitatively (median): {}".format(cinterval_median))
        boot_mean = self.bootstrap(data1, data2, n=5000, func=np.mean)
        cinterval_mean = boot_mean(.95)
        print("Quantitatively (mean): {}".format(cinterval_mean))
        
    
    ## Calculate Correlation
    def calculateCorrelation(self, data, method):
        col_names = list(data.columns)
        print(col_names)
        for i in range(len(col_names)):
            for j in range(i+1, len(col_names)):
                if method == 'spearman':
                    corr, p_val = spearmanr(data[col_names[i]], data[col_names[j]])
                    print('{} - {}: corr:{}, pval:{}'.format(col_names[i], col_names[j], corr, p_val))
                elif method == 'pearson':
                    corr, p_val = pearsonr(data[col_names[i]], data[col_names[j]])
                    print('{} - {}: corr:{}, pval:{}'.format(col_names[i], col_names[j], corr, p_val))
    
    ## Calculate Partial Correlation
    def partial_corr(self, data, method):
        col_names = list(data.columns)
        print(col_names)
        for i in range(len(col_names)):
            for j in range(i+1, len(col_names)):
                print(col_names[i], '--', col_names[j])
                covars = [covar for covar in col_names if covar!=col_names[i] and covar!=col_names[j]]
                if method == 'spearman':
                    print(pg.partial_corr(data=data, x=col_names[i], y=col_names[j], covar=covars, method='spearman').round(4))
                elif method == 'pearson':
                    print(pg.partial_corr(data=data, x=col_names[i], y=col_names[j], covar=covars, method='pearson').round(4))
            
    
    ###################################################################################
    ## PLOT ###########################################################################
    ###################################################################################
    
    ## plot simple line chart
    def plotLineChart(self, counts, y_label, x_label, save=False):
        plt.rcParams['figure.dpi'] = 400
        #x = range(1, len(counts) + 1, 1)
        x = [i.split(" ")[0] for i in self.aggregated_dates]
        fig, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(x, counts)
        ax1.plot(x, counts, marker='', color='green', linewidth=1, linestyle='solid', label="")
        ax1.legend(loc='upper left')
        ax1.set_xlabel("Week", fontsize=12)
        ax1.set_ylabel("#tweets", fontsize=12)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
        plt.title('Topic: ' + self.campaign.upper())
        if save:
            plt.savefig(self.campaign + '_' + y_label + ".png", bbox_inches="tight", pad_inches=0, format='png', dpi=400)
        plt.show()
    
    ## data_type: ['tweet' | 'video']
    def plotLineCharts(self, list_of_online_counts, data_type):
        plt.rcParams['figure.dpi'] = 400
        x = range(1, len(list_of_online_counts[0]) + 1, 1)
        if data_type == 'tweet':
            plt.plot(x, list_of_online_counts[0], marker='', color='green', linewidth=1, linestyle='dashed', label="Explicitly relevant tweets")
            plt.plot(x, list_of_online_counts[1], marker='', color='red', linewidth=1, linestyle='dashed', label="Implicitly relevant tweets")
            plt.ylabel("#tweets")
        elif data_type == 'video':
            plt.plot(x, list_of_online_counts[0], marker='', color='green', linewidth=1, linestyle='dashed', label="Explicitly relevant videos")
            plt.plot(x, list_of_online_counts[1], marker='', color='red', linewidth=1, linestyle='dashed', label="Implicitly relevant videos")
            plt.ylabel("#videos")
        plt.xlabel('Weeks')
        plt.title('Topic: ' + self.campaign.upper())
        plt.legend(loc='upper right')
        plt.show()

    def plotAreaChart(self, list_of_online_counts):
        x = range(1, len(list_of_online_counts[0]) + 1, 1)
        pal = sns.color_palette("Set1")
        plt.stackplot(x, list_of_online_counts, labels=['Explicitly relevant tweets','Union of relevant tweets'], colors=pal, alpha=0.4)
        plt.legend(loc='upper left')
        plt.ylabel('# of tweets')
        plt.xlabel('Weeks')
        plt.title('Topic: ' + self.campaign.upper())
        plt.show()

    ## data_type: ['tweet' | 'video']
    def plotTweetVideoChart(self, tweet_counts, video_counts):
        plt.rcParams['figure.dpi'] = 400
        x = [i.split(" ")[0] for i in self.aggregated_dates]
        #x = range(1, len(list_of_online_counts[0]) + 1, 1)
        fig, ax1 = plt.subplots(figsize=(12,4))
        #pal = sns.color_palette("Set1")
        ax2 = ax1.twinx()
        lns1 = ax1.plot(x, tweet_counts, marker='', color='green', linewidth=1.5, linestyle='dashed', label="tweets")
        lns2 = ax2.plot(x, video_counts, marker='', color='red', linewidth=1.5, linestyle=':', label="videos")
        ax1.set_xlabel('weeks', fontsize=14)
        ax1.set_ylabel("#tweets", color='green', fontsize=14)
        ax2.set_ylabel("#videos", color='red', fontsize=14)
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        #ax1.legend(loc='upper left')
        #ax2.legend(loc=7)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
        plt.title('Topic: ' + self.campaign.upper())
        plt.show()
    
    ## data_type: ['tweet' | 'video']
    def plotOnlineOfflineChart(self, list_of_online_counts, offline_counts, data_type, y2_label):
        x = range(1, len(list_of_online_counts[0]) + 1, 1)
        pal = sns.color_palette("Set1")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        if data_type == 'tweet':
            ax1.stackplot(x, list_of_online_counts, labels=['Explicitly relevant tweets','Union of relevant tweets'], colors=['green', 'blue'], alpha=0.4)
            ax1.set_ylabel("# of tweets", fontsize=14)
        elif data_type == 'video':
            ax1.stackplot(x, list_of_online_counts, labels=['Explicitly relevant videos','Union of relevant videos'], colors=['green', 'blue'], alpha=0.4)
            ax1.set_ylabel("# of videos", fontsize=14)
        ax2.plot(x, offline_counts, linewidth=1, color='darkorange', linestyle='dashed')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Weeks')
        ax2.set_ylabel(y2_label, color='darkorange', fontsize=14)
        plt.title('Topic: ' + self.campaign.upper())
        plt.show()

    ## data_type: ['tweet' | 'video']
    def plotOnlineOfflineChart2(self, list_of_online_counts, offline_counts, data_type, y1_label, y2_label, dates):
        #x = range(1, len(list_of_online_counts[0]) + 1, 1)
        x = [i.split(" ")[0] for i in dates]
        pal = sns.color_palette("Set1")
        fig, ax1 = plt.subplots(figsize=(12,4))
        #plt.figure(figsize=(20,10))
        if offline_counts != None:
            ax2 = ax1.twinx()
        if data_type == 'tweet':
            ax1.plot(x, list_of_online_counts[0], marker='', color='green', linewidth=1, linestyle='solid', label="Explicitly relevant tweets")
            #ax1.plot(x, list_of_online_counts[1], marker='', color='red', linewidth=1, linestyle='solid', label="Implicitly relevant tweets")
            ax1.set_ylabel(y1_label, fontsize=12)
        elif data_type == 'video':
            #ax1.plot(x, list_of_online_counts[0], marker='', color='green', linewidth=1, linestyle='solid', label="Explicitly relevant videos")
            ax1.plot(x, list_of_online_counts[1], marker='', color='red', linewidth=1, linestyle='solid', label="Implicitly relevant videos")
            #ax1.plot(x, list_of_online_counts[2], marker='', color='black', linewidth=1, linestyle='solid', label="Videos referred by tweets")
            ax1.set_ylabel(y1_label, fontsize=12)
        
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Weeks', fontsize=12)
        if offline_counts != None:
            ax2.plot(x, offline_counts, linewidth=1, color='blue', linestyle='dotted', label="# of affected")
            ax2.set_ylabel(y2_label, color='blue', fontsize=12)
            ax2.legend(loc='upper right')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
        plt.title('Topic: ' + self.campaign.upper())
        plt.savefig(data_type + ".png", bbox_inches="tight", pad_inches=0, format='png', dpi=1000)
        plt.show()
    
    ## plot percentile chart
    def plotPercentileChart(self, values, y_label, x_label, title):
        # Percentile values
        #p = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        values = np.array(values)
        values.sort()
        plt.figure(figsize=(12,4))
        p = np.arange(0, 101, 5.0)
        #perc = mlab.prctile(values, p=p)
        perc = np.percentile(np.cumsum(values), p)
        plt.plot(np.cumsum(values))
        # Place red dots on the percentiles
        plt.plot((len(values)-1) * p/100., perc, 'ro')
        # Set tick locations and labels
        plt.xticks((len(values)-1) * p/100., map(str, p))
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel(x_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.show()
    
    ## plot percentile chart in log10
    def plotLogPercentileChart(self, values, y_label, x_label, title):
        # Percentile values
        #p = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        values = np.array(values)
        values.sort()
        plt.figure(figsize=(6,4))
        p = np.arange(0, 101, 10.0)
        perc = np.percentile(np.cumsum(values), p)
        perc_log = np.log10(perc)
        plt.plot(np.log10(np.cumsum(values)))
        #plt.plot(np.cumsum(values))
        print("perc: ", perc)
        print("perc_log: ", perc_log)
        # Place red dots on the percentiles
        plt.plot((len(values)-1) * p/100., perc_log, 'ro')
        print(p/100.)
        print((len(values)-1) * p/100.)
        # Set tick locations and labels
        plt.xticks((len(values)-1) * p/100., map(str, p))
        
        '''
        plt.figure(figsize=(12,4))
        p = 1. * np.arange(len(values)) / (len(values) - 1)
        # plot the sorted data:
        plt.plot(p, np.log10(np.cumsum(values)))
        '''
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel(x_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.show()
    
    
    ## plot percentile chart in log10
    def plotLogPercentileChart2(self, list_of_values, y_label, x_label, title):
        # Percentile values
        plt.figure(figsize=(6,4))
        p = np.arange(0, 101, 10.0)
        for values in list_of_values:
            values = np.array(values)
            values.sort()
            #perc = np.percentile(np.log10(np.cumsum(values)), p)
            perc = np.percentile(np.log10(values), p)
            plt.plot(p, perc)
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel(x_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.show()
    
    ## plot CDF for a single variable in log10
    def plotCDFSingle(self, values, x_label, title):
        color = "orange"
        plt.rcParams['figure.dpi'] = 400
        values = np.array(values)
        values.sort()
        #ecdf = ECDF(values)
        ecdf = ECDF(np.log10(values))
        #ecdf = ECDF(np.log10(np.cumsum(values)))
        ecdf = ECDF(np.log10(values))
        plt.plot(ecdf.x, ecdf.y, color=color, linestyle="--", linewidth=1.5)
        plt.ylabel("CDF", fontsize=12)
        plt.xlabel("{} (log10)".format(x_label), fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        plt.show()
    
    ## plot CDF for multiple variables
    def plotCDFMultiple(self, list_of_values, list_of_labels, x_label, title, log_based):
        colors = ["blue", "red", "orange", "purple", "green", "grey"]
        plt.rcParams['figure.dpi'] = 400
        for idx in range(len(list_of_values)):
            values = np.array(list_of_values[idx])
            values.sort()
            if log_based:
                ecdf = ECDF(np.log10(values))
            else:
                ecdf = ECDF(values)
            #ecdf = ECDF(np.log10(np.cumsum(values)))
            plt.plot(ecdf.x, ecdf.y, color=colors[idx], label=list_of_labels[idx], linestyle="--", linewidth=1.5)
        plt.ylabel("CDF", fontsize=12)
        if log_based:
            plt.xlabel("{} (log10)".format(x_label), fontsize=12)
        else:
            plt.xlabel("{}".format(x_label), fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        plt.show()
    
    ## plot Seaborn Histogram-KDE for multiple variables
    def plotHistKDEMulitple(self, list_of_values, list_of_labels, x_label, title, log_based):
        colors = ["blue", "red", "orange", "purple", "green", "grey"]
        plt.rcParams['figure.dpi'] = 400
        for idx in range(len(list_of_values)):
            if log_based:
                sns.distplot(np.log10(list_of_values[idx]), label=list_of_labels[idx], color=colors[idx])
            else:
                sns.distplot(list_of_values[idx], label=list_of_labels[idx], color=colors[idx])
        plt.ylabel("KDE", fontsize=12)
        if log_based:
            plt.xlabel("{} (log10)".format(x_label), fontsize=12)
        else:
            plt.xlabel("{}".format(x_label), fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        plt.show()
    
    ## Plot Scatter plot with different Marker size
    def plotScatter(self, x, y, marker_sizes, x_label, y_label, title, log_based, color):
        if log_based:
            x = np.log10(x)
            y = np.log10(y)
        plt.rcParams['figure.dpi'] = 400
        if color != None and type(color) == list and type(color[0]) == float:
            plt.scatter(x, y, s=marker_sizes, marker="o", alpha=0.75, linewidths=0.5, c=color, cmap='RdYlGn_r')
        else:
            plt.scatter(x, y, s=marker_sizes, marker="o", alpha=0.5, linewidths=0.5, color=color)
        xpoints = ypoints = plt.ylim()
        plt.plot(xpoints, ypoints, linestyle='--', color='gray', lw=1, alpha=0.25, scalex=False, scaley=False)
        if log_based:
            plt.xlabel("{} (log10)".format(x_label), fontsize=12)
            plt.ylabel("{} (log10)".format(y_label), fontsize=12)
        else:
            plt.xlabel("{}".format(x_label), fontsize=12)
            plt.ylabel(y_label, fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        
        #axes = plt.gca()
        #axis_max = max(np.amax(x), np.amax(y))
        #axes.set_xlim([0,axis_max])
        #axes.set_ylim([0,axis_max])
        plt.show()
    
    ## Plot Scatter plot with different Marker size
    def plotScatterWithLabel(self, x, y, marker_sizes, marker_labels, x_label, y_label, title, log_based, color):
        if log_based:
            x = np.log10(x)
            y = np.log10(y)
        plt.rcParams['figure.dpi'] = 400
        
        fig, ax = plt.subplots()
        
        if color != None and type(color) == list and type(color[0]) == float:
            ax.scatter(x, y, s=marker_sizes, marker="o", alpha=0.75, linewidths=0.5, c=color, cmap='RdYlGn_r')
        else:
            ax.scatter(x, y, s=marker_sizes, marker="o", alpha=0.5, linewidths=0.5, color=color)
        #xpoints = ypoints = plt.ylim()
        #ax.plot(xpoints, ypoints, linestyle='--', color='gray', lw=1, alpha=0.25, scalex=False, scaley=False)
        if log_based:
            plt.xlabel("{} (log10)".format(x_label), fontsize=12)
            plt.ylabel("{} (log10)".format(y_label), fontsize=12)
        else:
            plt.xlabel("{}".format(x_label), fontsize=12)
            plt.ylabel(y_label, fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        
        for i, txt in enumerate(marker_labels):
            ax.annotate(txt, (x[i], y[i]), fontsize=5.5)
        
        from sklearn.linear_model import LinearRegression
        X = np.array([[item] for item in x])
        y = np.array(y)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        #print(y_pred)
        plt.plot(X, y_pred, color='red')
        
        #axes = plt.gca()
        #axis_max = max(np.amax(x), np.amax(y))
        #axes.set_xlim([0,axis_max])
        #axes.set_ylim([0,axis_max])
        plt.show()
    
    ## plot Lorenz Curve
    def plotLorenzCurve(self, list_of_values, y_label, title):
        list_of_values = np.array(list_of_values)
        scaled_prefix_sum = list_of_values.cumsum() / list_of_values.sum()
        lorenz_curve = np.insert(scaled_prefix_sum, 0, 0)
        # we need the X values to be between 0.0 to 1.0
        plt.rcParams['figure.dpi'] = 400
        plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)
        # plot the straight line perfect equality curve
        plt.plot([0,1], [0,1])
        plt.ylabel("% of {}".format(y_label), fontsize=12)
        plt.xlabel("% of users", fontsize=12)
        if title != None and title != "":
            plt.title(title, fontsize=14)
        plt.legend(prop={'size': 10})
        plt.show()
    
    ## convert Youtube dureation from PTMS format to minutes
    def convertDurationToSeconds(self, date):
        date_values = list(map(int, re.findall(r'\d+', date)))
        date_types = []
        if 'H' in date:
            date_types.append('H')
        if 'M' in date:
            date_types.append('M')
        if 'S' in date:
            date_types.append('S')
        
        total = 0.
        for i in range(len(date_types)):
            if date_types[i] == 'H':
                total += (60* 60 * date_values[i])
            elif date_types[i] == 'M':
                total += (60 * date_values[i])
            elif date_types[i] == 'S':
                total += date_values[i]
        
        return total
    
    ## min max scaling of given array
    def scaleMinMax(self, values, min_val, max_val):
        values = np.array(values)
        new_values = (((values - np.amin(values)) / (np.amax(values)-np.amin(values))) * (max_val-min_val)) + min_val
        return list(new_values)
    
    ## corr between x and shifted (lagged) copies of a vector y 
    def timeLaggedCorr(self, x, y, max_lag=20):
        for lag in range(-max_lag, max_lag+1):
            shifted_y = np.roll(y, lag)
            corr, p_val = spearmanr(x, shifted_y)
            print('lag %d corr %.3f' % (lag,corr), "p-val: %.4f" % p_val)
    
    
    
    
    ###################################################################################
    ## SOME RARE OPERATIONS ###########################################################
    ###################################################################################
    ## create candidate Videos CSV file for manual annotation if they are topic-relevant or not.
    def createCandidateVideosFile(self):
        all_tweets = pickle.load(open('data/from_anu/v1/{}_tweets.pkl'.format(self.campaign), 'rb'))
        videos = pickle.load(open('data/from_anu/v1/{}_videos.pkl'.format(self.campaign), 'rb'))
        active_videos = pickle.load(open('data/from_anu/_active_videos.pkl', 'rb'))
        
        candidate_videos = {}
        for tweet_id in all_tweets:
            user_id = all_tweets[tweet_id]['_source']['user_id_str']
            original_video_ids = all_tweets[tweet_id]['_source']['original_vids'].split(';')
            retweeted_video_ids = all_tweets[tweet_id]['_source']['retweeted_vids'].split(';')
            quoted_video_ids = all_tweets[tweet_id]['_source']['quoted_vids'].split(';')
            video_ids = list(set(original_video_ids + retweeted_video_ids + quoted_video_ids))
            if 'N' in video_ids:
                video_ids.remove('N')
            for video_id in video_ids:
                if video_id not in candidate_videos and video_id in active_videos:
                    candidate_videos[video_id] = ['twitter']
        for video_id in videos:
            if video_id not in candidate_videos:
                candidate_videos[video_id] = ['youtube']
            else:
                candidate_videos[video_id].append('youtube')
        print('all:', len(candidate_videos.keys()))
        
        twitter = 0
        youtube = 0
        both = 0
        for video_id in candidate_videos:
            if len(candidate_videos[video_id]) == 1:
                if candidate_videos[video_id][0] == 'twitter':
                    twitter += 1
                elif candidate_videos[video_id][0] == 'youtube':
                    youtube += 1
            elif len(candidate_videos[video_id]) == 2:
                both += 1
        
        print('twitter:', twitter, 'youtube:', youtube, 'both:', both)
        with open('{}_all_candiate_videos.csv'.format(self.campaign), 'w') as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            for video_id in candidate_videos:
                video_link = "https://www.youtube.com/watch?v={}".format(video_id)
                writer.writerow([video_id, video_link, ','.join(candidate_videos[video_id])])
    
    ## Create video files to send Siqi to annotate their subtitle information.
    def createVideosFileForManualAnnotations(self):
        videos = pickle.load(open('data/from_anu/v1/{}_all_videos_from_candidate_videos.pkl'.format(self.campaign), 'rb'))
                
        candidate_videos_t = {}
        candidate_videos_ty = {}
        candidate_videos_t_ty = {}
        c_music = 0
        c_other_lang = 0
        c_all = 0
        c_suitable = 0
        aa = 0
        with open('data/from_anu/v1/{}_all_candiate_videos.csv'.format(self.campaign), 'r') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            for row in reader:
                vid = row[0]
                video_link = row[1]
                source = row[2]
                '''
                if source == 'twitter'
                candidate_videos[video_id] = {'source': source, 'link': video_link}
                '''
                lang = None
                is_music = False
                if 'defaultLanguage' in videos[vid]['_source']['snippet']:
                    lang = videos[vid]['_source']['snippet']['defaultLanguage']
                elif 'defaultAudioLanguage' in videos[vid]['_source']['snippet']:
                    lang = videos[vid]['_source']['snippet']['defaultAudioLanguage']
                elif 'detectLanguage' in videos[vid]['_source']['snippet']:
                    lang = videos[vid]['_source']['snippet']['detectLanguage']
                if lang != None and (lang!='en' and lang!='en-GB' and lang!='en-US'):
                    c_other_lang += 1

                if 'topicDetails' in videos[vid]['_source']:
                    topics = [topic.split('/')[-1] for topic in videos[vid]['_source']['topicDetails']['topicCategories']]
                    c = 0
                    for topic in topics:
                        if 'music' in topic.lower():
                            c+=1
                    if c == len(topics):
                        is_music = True
                        c_music+=1

                if source == 'twitter,youtube':
                    aa+=1

                if (lang==None or lang=='en' or lang=='en-GB' or lang=='en-US') and (is_music==False):
                    #print(vid, lang, is_music)
                    if source == 'twitter':
                        candidate_videos_t[vid] = {'link': video_link, 'source': source}
                    elif source == 'twitter,youtube':
                        candidate_videos_ty[vid] = {'link': video_link, 'source': source}
                    if source == 'twitter' or source == 'twitter,youtube':
                        candidate_videos_t_ty[vid] = {'link': video_link, 'source': source}
                    c_suitable+=1

                c_all+=1
            
            print("c_all: {}".format(c_all))
            print("c_other_lang: {}, c_music: {}".format(c_other_lang, c_music))
            print("c_suitable: {}".format(c_suitable))
            
            with open('{}_all_videos_for_subtitle_annotation_t.csv'.format(self.campaign), 'w') as f:
                writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL, escapechar='\\')
                for vid in candidate_videos_t:
                    writer.writerow([vid, candidate_videos_t[vid]['link'], candidate_videos_t[vid]['source']])
            with open('{}_all_videos_for_subtitle_annotation_ty.csv'.format(self.campaign), 'w') as g:
                writer = csv.writer(g, delimiter='\t', quoting=csv.QUOTE_ALL, escapechar='\\')
                for vid in candidate_videos_ty:
                    writer.writerow([vid, candidate_videos_ty[vid]['link'], candidate_videos_ty[vid]['source']])
            with open('{}_all_videos_for_subtitle_annotation_t_ty.csv'.format(self.campaign), 'w') as h:
                writer = csv.writer(h, delimiter='\t', quoting=csv.QUOTE_ALL, escapechar='\\')
                for vid in candidate_videos_t_ty:
                    writer.writerow([vid, candidate_videos_t_ty[vid]['link'], candidate_videos_t_ty[vid]['source']])
            
            
            print(len(candidate_videos_t.keys()))
            print(len(candidate_videos_ty.keys()))
            print(len(candidate_videos_t_ty.keys()))
            print(aa)  
    
    ## create Videos CSV file for manual annotation if they are topic-relevant or not considerg subtitles.
    def createVideosFileForManualAnnotationsUsingSubtitles(self):
        subtitle_file_path = "data/from_anu/v1/{}_subtitle_annotations.csv".format(self.campaign)
        counter = 0
        c_t = 0
        c_ty = 0
        videos_t = {}
        videos_ty = {}
        with open(subtitle_file_path,  newline='', mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                vid = row[0]
                video_link = row[1]
                source = row[2]
                subtitle_match = row[3]
                if source == 'twitter' and subtitle_match == '1':
                    matched_kws = row[4] 
                    videos_t[vid] = {'link': video_link, 'source': source, 'subtitle_match': subtitle_match, 'matched_kws': matched_kws}
                    c_t+=1
                elif source == 'twitter,youtube':
                    videos_ty[vid] = {'link': video_link, 'source': source, 'subtitle_match': subtitle_match}
                    c_ty+=1
                counter+=1
        
        
        with open('{}_all_videos_for_annotation_twitter.csv'.format(self.campaign), 'w') as g:
            writer = csv.writer(g, delimiter='\t', quoting=csv.QUOTE_ALL, escapechar='\\')
            for vid in videos_t:
                writer.writerow([vid, videos_t[vid]['link'], videos_t[vid]['source']])
        
        
        with open('{}_all_videos_for_annotation_twitter_youtube.csv'.format(self.campaign), 'w') as h:
            writer = csv.writer(h, delimiter='\t', quoting=csv.QUOTE_ALL, escapechar='\\')
            for vid in videos_ty:
                writer.writerow([vid, videos_ty[vid]['link'], videos_ty[vid]['source'], videos_ty[vid]['subtitle_match']])
        
        print('#vids in subtitle_annotations:', counter)
        print('#vids for annotation (Twitter):', len(videos_t.keys()))
        print('#vids for automatically relevant (Twitter,Youtube):', len(videos_ty.keys()))
                        
    
    ## check compliance of annotated videos with youtube search results.
    def checkComplianceWithYoutubeSearch(self):
        videos_youtube_search = []
        with open('data/from_anu/v1/{}_videos_youtube_search.json'.format(self.campaign), 'r') as f:
            for line in f:
                videos_youtube_search.append(json.loads(line))
        #print(videos_youtube_search)
        videos_youtube_search_ids = set([item['id'] for item in videos_youtube_search])
        
        relevant_video_ids = []
        irrelevant_video_ids = []
        na_video_ids = []
        with open('data/from_anu/v1/{}_video_annotations.csv'.format(self.campaign)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                video_id = row[0]
                label = row[3]
                if label == '1':
                    relevant_video_ids.append(video_id)
                elif label == '0':
                    irrelevant_video_ids.append(video_id)
                elif label == 'NA':
                    na_video_ids.append(video_id)
                line_count += 1
        
        diff_1 = videos_youtube_search_ids.difference(set(relevant_video_ids + irrelevant_video_ids + na_video_ids))
        diff_2 = set(relevant_video_ids + irrelevant_video_ids + na_video_ids).difference(videos_youtube_search_ids)
        
        common_relevant = set(relevant_video_ids).intersection(videos_youtube_search_ids)
        common_irrelevant = set(irrelevant_video_ids).intersection(videos_youtube_search_ids)
        common_na = set(na_video_ids).intersection(videos_youtube_search_ids)
        
        print('youtube_search:', len(set(videos_youtube_search_ids)))
        print('annotated_videos:', len(set(relevant_video_ids + irrelevant_video_ids + na_video_ids)))
        print('youtube_search / annotated_videos:', len(diff_1))
        print('annotated_videos / youtube_search:', len(diff_2))
        print('common_relevant:', len(common_relevant))
        print('common_irrelevant:', len(common_irrelevant))
        print('common_na:', len(common_na))
        
        print(diff_1)