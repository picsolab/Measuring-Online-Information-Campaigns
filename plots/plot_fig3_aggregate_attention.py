# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot aggregate attention comparison on YouTube and Twitter
Usage: python plot_fig3_aggregate_attention.py
Input data file: ../data/abo_videos.json, ../data/gun_videos.json, ../data/blm_videos.json
Output image file: ../images/view120.pdf, ../images/relative_engagement.pdf, ../images/likes.pdf,
                   ../images/viral_potential.pdf, ../images/tweet120.pdf
Time: ~1M
"""

import sys, os, json
import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.helper import hide_spines

rc = {'axes.labelsize': 22, 'legend.fontsize': 22,
      'axes.titlesize': 22, 'xtick.labelsize': 22, 'ytick.labelsize': 22}
sns.set(rc=rc)
sns.set_style(style='white')

plt.rcParams['figure.dpi'] = 600


def exponent_fmt(x, pos):
    """ The two args are the value and tick position. """
    if x == 0:
        return '1'
    else:
        return '$10^{{{0:.0f}}}$'.format(x)


def main():
    topic_list = []
    party_list = []
    view_120_list = []
    engagement_list = []
    virality_list = []
    like_list = []
    tweet_120_list = []

    plot_view120 = True
    plot_engagement = True
    plot_like = True
    plot_virality = True
    plot_tweet120 = True

    for topic, filename in zip(['Abortion', 'Gun control', 'BLM'], ['../data/abo_videos.json', '../data/gun_videos.json', '../data/blm_videos.json']):
        with open(filename, 'r') as fin:
            for line in fin:
                video_json = json.loads(line.rstrip())
                political_label = video_json['political_label']
                if political_label == 'L' or political_label == 'R':
                    topic_list.append(topic)

                    view_120_list.append(np.log10(video_json['total_view120']))
                    engagement_list.append(video_json['relative_engagement'])
                    like_list.append(video_json['frac_like'])
                    virality_list.append(np.log10(video_json['virality']))
                    tweet_120_list.append(np.log10(video_json['total_tweet120']))

                    if political_label == 'L':
                        party_list.append('Left')
                    elif political_label == 'R':
                        party_list.append('Right')

    df = pd.DataFrame({'topic': topic_list, 'party': party_list,
                       'view_120_list': view_120_list,
                       'engagement_list': engagement_list,
                       'like_list': like_list,
                       'virality_list': virality_list,
                       'tweet_120_list': tweet_120_list})

    print(df.groupby(['topic', 'party']).median())

    for topic in ['Abortion', 'Gun control', 'BLM']:
        for metric in ['view_120_list', 'engagement_list', 'like_list', 'virality_list', 'tweet_120_list']:
            left = df[(df.topic == topic) & (df.party == 'Left')][metric]
            right = df[(df.topic == topic) & (df.party == 'Right')][metric]
            print(topic, metric)
            print(pg.mwu(left, right, tail='one-sided'))

    if plot_view120:
        ax = sns.violinplot(x=df["topic"], y=df['view_120_list'], hue=df["party"],
                            palette={"Right": "#e06666", "Left": "#6d9eeb"},
                            inner="quartile",
                            linewidth=1.5, cut=1.5,
                            scale="area", split=True, width=0.75, hue_order=['Left', 'Right'])

        ax.set(xlabel=None)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], loc='lower center', frameon=False)
        ax.set_ylabel('views at day 120')
        ax.set_yticks([3, 5, 7])
        ax.yaxis.set_major_formatter(FuncFormatter(exponent_fmt))
        hide_spines(ax)
        ax.set_title('(a)', pad=-4.3 * 72, y=1)
        plt.savefig("../images/view120.pdf", bbox_inches='tight', dpi=600)
        plt.clf()

    if plot_engagement:
        ax = sns.violinplot(x=df["topic"], y=df['engagement_list'], hue=df["party"],
                            palette={"Right": "#e06666", "Left": "#6d9eeb"},
                            inner="quartile",
                            linewidth=1.5, cut=0,
                            scale="area", split=True, width=0.75, hue_order=['Left', 'Right'])

        ax.set(xlabel=None)
        ax.get_legend().remove()
        ax.set_ylabel('relative engagement')
        hide_spines(ax)
        ax.set_title('(b)', pad=-4.3 * 72, y=1)
        plt.savefig("../images/relative_engagement.pdf", bbox_inches='tight', dpi=600)
        plt.clf()

    if plot_like:
        ax = sns.violinplot(x=df["topic"], y=df['like_list'], hue=df["party"],
                            palette={"Right": "#e06666", "Left": "#6d9eeb"},
                            inner="quartile",
                            linewidth=1.5, cut=0.5,
                            scale="area", split=True, width=0.75, hue_order=['Left', 'Right'])

        ax.set(xlabel=None)
        ax.get_legend().remove()
        ax.set_yticks([0, 1])
        ax.set_ylabel('fraction of likes')
        hide_spines(ax)
        ax.set_title('(c)', pad=-4.3 * 72, y=1)
        plt.savefig("../images/likes.pdf", bbox_inches='tight', dpi=600)
        plt.clf()

    if plot_virality:
        ax = sns.violinplot(x=df["topic"], y=df['virality_list'], hue=df["party"],
                            palette={"Right": "#e06666", "Left": "#6d9eeb"},
                            inner="quartile",
                            linewidth=1.5, cut=1.5,
                            scale="area", split=True, width=0.75, hue_order=['Left', 'Right'])

        ax.set(xlabel=None)
        ax.get_legend().remove()
        ax.yaxis.set_major_formatter(FuncFormatter(exponent_fmt))
        ax.set_yticks([-2, 0, 2, 4])
        ax.set_ylabel('viral potential')
        hide_spines(ax)
        ax.set_title('(d)', pad=-4.3 * 72, y=1)
        plt.savefig("../images/viral_potential.pdf", bbox_inches='tight', dpi=600)
        plt.clf()

    if plot_tweet120:
        ax = sns.violinplot(x=df["topic"], y=df['tweet_120_list'], hue=df["party"],
                            palette={"Right": "#e06666", "Left": "#6d9eeb"},
                            inner="quartile",
                            linewidth=1.5, cut=0,
                            scale="area", split=True, width=0.75, hue_order=['Left', 'Right'])

        ax.set(xlabel=None)
        ax.get_legend().remove()
        ax.set_ylabel('tweets at day 120')
        ax.set_yticks([2, 3, 4])
        ax.yaxis.set_major_formatter(FuncFormatter(exponent_fmt))
        hide_spines(ax)
        ax.set_title('(e)', pad=-4.3 * 72, y=1)
        plt.savefig("../images/tweet120.pdf", bbox_inches='tight', dpi=600)
        plt.clf()


if __name__ == '__main__':
    main()
