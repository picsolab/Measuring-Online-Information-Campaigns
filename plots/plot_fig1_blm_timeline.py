# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the attention time series related to BLM throughout 2017.
Usage: python plot_fig1_blm_timeline.py
Input data file: ../data/blm_videos.json
Output image file: ../images/blm_annotated_timeline.pdf
Time: ~1M
"""

import sys, os, json
from datetime import datetime, timedelta
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

from utils.helper import str2obj, concise_fmt


def plot_timeline(axes):
    num_day = (datetime(2018, 4, 30) - datetime(2017, 1, 1)).days + 1
    x_axis = [datetime(2017, 1, 1).date() + timedelta(days=x) for x in range(num_day)]

    left_view_timeline = np.zeros(num_day)
    right_view_timeline = np.zeros(num_day)
    left_tweet_timeline = np.zeros(num_day)
    right_tweet_timeline = np.zeros(num_day)

    num_left_video = 0
    num_left_view = 0
    num_left_tweet = 0

    num_left_colbert_video = 0
    num_left_colbert_view = 0
    num_left_colbert_tweet = 0

    num_right_video = 0
    num_right_view = 0
    num_right_tweet = 0

    with open('../data/blm_videos.json', 'r') as fin:
        for line in fin:
            video_json = json.loads(line.rstrip())
            start_date = video_json['start_date']
            start_date = str2obj(start_date, fmt='youtube')
            day_gap = (start_date - datetime(2017, 1, 1)).days
            daily_view120 = np.array(video_json['daily_view120'])
            daily_tweet120 = np.array(video_json['daily_tweet120'])
            channel_id = video_json['channel_id']
            if video_json['political_label'] == 'L':
                left_view_timeline[day_gap: day_gap + 120] += daily_view120
                left_tweet_timeline[day_gap: day_gap + 120] += daily_tweet120

                if str2obj('2017-08-11', fmt='youtube') <= start_date <= str2obj('2017-08-25', fmt='youtube'):
                    num_left_video += 1
                    num_left_view += sum(daily_view120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])
                    num_left_tweet += sum(daily_tweet120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])

                    if channel_id == 'UCMtFAi84ehTSYSE9XoHefig':
                        num_left_colbert_video += 1
                        print(start_date, video_json['title'])
                        num_left_colbert_view += sum(daily_view120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])
                        num_left_colbert_tweet += sum(daily_tweet120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])

            elif video_json['political_label'] == 'R':
                right_view_timeline[day_gap: day_gap + 120] += daily_view120
                right_tweet_timeline[day_gap: day_gap + 120] += daily_tweet120

                if str2obj('2017-08-11', fmt='youtube') <= start_date <= str2obj('2017-08-25', fmt='youtube'):
                    num_right_video += 1
                    num_right_view += sum(daily_view120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])
                    num_right_tweet += sum(daily_tweet120[: (start_date - str2obj('2017-08-11', fmt='youtube')).days + 1])

    axes[0].plot_date(x_axis[: 365], left_view_timeline[: 365], '-', c='b', label='Left', zorder=50)
    axes[0].plot_date(x_axis[: 365], right_view_timeline[: 365], '-', c='r', label='Right', zorder=50)

    axes[1].plot_date(x_axis[: 365], left_tweet_timeline[: 365], '-', c='b', label='Left', zorder=50)
    axes[1].plot_date(x_axis[: 365], right_tweet_timeline[: 365], '-', c='r', label='Right', zorder=50)

    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    axes[1].set_xlabel('date', fontsize=9)
    axes[0].set_ylabel('daily view count', fontsize=9)
    axes[1].set_ylabel('daily tweet count', fontsize=9)

    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(concise_fmt))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.axvspan(datetime(2017, 8, 11), datetime(2017, 8, 25), alpha=0.5, color='grey', zorder=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    print('\nwithin target period')
    print('{0:,} left videos vs. {1:,} right videos'.format(num_left_video, num_right_video))
    print('{0:,} left views vs. {1:,} right views'.format(num_left_view, num_right_view))
    print('{0:,} left tweets vs. {1:,} right tweets'.format(num_left_tweet, num_right_tweet))

    print('Stephen Colbert contributed {0} videos, {1} views, {2} tweets'.format(num_left_colbert_video, num_left_colbert_view, num_left_colbert_tweet))


def main():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 3), sharex='col')
    axes = axes.ravel()

    plot_timeline(axes)

    plt.tight_layout()
    plt.savefig('../images/blm_annotated_timeline.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
