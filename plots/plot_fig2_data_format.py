# !/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the daily view count series and tweet cascades for an example video.
Usage: python plot_fig2_data_format.py
Input data file: ../data/abo_videos.json
Output image file: ../images/yt_tw_data_format.pdf
Time: ~1M
"""

import sys, os, json
from datetime import datetime, timedelta
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.helper import str2obj, melt_snowflake, concise_fmt, hide_spines

rc = {'axes.titlesize': 22, 'axes.labelsize': 18, 'legend.fontsize': 18,
      'font.size': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(rc)

tweets = [{'tid': '933934351133302784', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933574103910330368', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '936673271961055233', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933732734299602945', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933516227154235392', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933590700180103168', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933448824701546497', 'type': 'root', 'children': ['933450543028109313', '933457715598450688', '933453479103942658', '933452401150902274', '934445043481501696', '933449606532358145', '933461278294605825', '933451123511324672', '933482040732934146', '933452379466358784', '933504435199922177', '933454218152865792', '933449088506519552', '933449333764247555'], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933694647452487680', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933496087788351488', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933478101153087488', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933453923771351041', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933448598649556992', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933454705002459136', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '958898619914047488', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933497193834663936', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933483790994984960', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933476342213038081', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933645897749225472', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933459631611006976', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933522693726580737', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933574901235101696', 'type': 'root', 'children': ['933576647873388544'], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933886139878043649', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933454404648304641', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933564517631561728', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933507890962468865', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '934475103630438400', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933741009225252864', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933568160237748224', 'type': 'root', 'children': ['937733789895163904', '936788867444703234', '939738570658418688', '934849964223578112', '942480501742247936', '941102442103562240', '933870433983643648', '935566102385405953', '937416635123507200', '934250539444924416'], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933827932144926720', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '935377425004990465', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933459097902616578', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '934115249866838017', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933453428705103872', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '934023859694718979', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933458428755881984', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933526478146416641', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933502832526835712', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933647384936054784', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '934589174484733953', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933821593209462784', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933497431790182400', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933464925367922689', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933457673563049984', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933448978468950016', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933458413509644288', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '945691752010219521', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933742526594535425', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933449658357071873', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933579104145719301', 'type': 'root', 'children': ['933699472156012544', '933579543679442944', '933753163886325760', '933584212942295041', '933634284983697409', '933585751832281089', '933580391184814080', '933580093514924032', '936018661046829057', '933725033477816320', '933658695472263168', '933579490755616769', '933579876036116481', '933748356106502144', '933797506600673280', '934061365341306880', '933767547471536128', '933734022194323457', '933580115358994433', '933583674817249281', '933698649170763782', '933581031025754113', '933579332425015296', '933726131424960512', '933600700336103425', '933608866696146944', '934444484020064256', '933581179915329536', '933579530484310016', '933585610744455169', '933580497225240576', '933581226790678528', '933734203404910592', '933927910213373955', '933580616465035264'], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933718119553658880', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933463415229644800', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933450446160650240', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933452058275074049', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933786624927952897', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}, {'tid': '933578908040970241', 'type': 'root', 'children': [], 'vid': 'PTjMngQwGh8;N;N'}]

with open('../data/abo_videos.json', 'r') as fin:
    for line in fin:
        video_json = json.loads(line.rstrip())
        video_id = video_json['video_id']
        if video_id == 'PTjMngQwGh8':
            start_date = video_json['start_date']
            start_date = str2obj(start_date, fmt='youtube')
            daily_view120 = video_json['daily_view120']

            x_axis = range(120)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex='col')

            x_axis = [start_date + timedelta(days=x) for x in x_axis]
            ax1.plot_date(x_axis, daily_view120, '-', c='r')
            ax1.set_yscale('symlog')

            ax1.yaxis.set_major_formatter(FuncFormatter(concise_fmt))
            ax1.set_yticks([0, 1, 100, 10000])

            ax1.set_ylabel('view count')

            x2 = []
            y2 = []

            total = 0
            for tweet in tweets:
                total += 1
                tid = tweet['tid']
                timestamp = melt_snowflake(tid)[0]/1000
                ts = datetime.utcfromtimestamp(timestamp)
                # ts = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')
                # print(ts)
                idx = len(tweet['children'])
                x2.append(ts)
                y2.append(idx)

                total += idx

            print(total)
            ax2.scatter(x2, np.array(y2) + 1, s=10 * np.array(y2) + 10, facecolors='none', edgecolors='b')

            ax2.set_xticks([datetime(2017, 11, 22), datetime(2017, 12, 22), datetime(2018, 1, 22)])
            ax2.set_ylabel('cascade size')
            ax2.set_ylim([-1, 40])

            ax1.set_xlim([datetime(2017, 11, 15), datetime(2018, 2, 5)])

            hide_spines((ax1, ax2))

            plt.tight_layout()
            plt.savefig('../images/yt_tw_data_format.pdf', bbox_inches='tight')
            plt.show()
