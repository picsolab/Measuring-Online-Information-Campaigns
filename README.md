## pipeline.pdf
Shows the pipline step by step including (1) how to extract topic relevant tweets and videos, (2) how to find early adopters and their shared audience network, (3) how to apply political leaning propagation, (4) how to find political leaning thresholds for videos, (5) how to extract and analyze the engagement, network structural, temporal, language, cascade and YouTube reaction information for YouTube videos using early adopters in Twitter.

# Folders

## src
Scripts and LIWC tool for text analysis. Since LIWC is licensed, this folder is only for internal use. Please **do not make this folder public.**

## elasticsearch
Scripts to retrieve data from ANU servers.

## twitter_data_collection
Scripts to perform data gathering from Twitter including checking users' existence, collecting users' followers-friends and creating the shared audience graph for the users.


# Notebooks and Python scripts

## tweet.py
Methods to extract information about early adopters' of a topic in terms of engagment, network structure, temporal, language, cascade and YouTube reaction. It also includes methods to extract information about user profiles including location and seed political leanings. 

## video.py
Methods to extract information about YouTube videos.

## utils.py
Methods for visualizaion and some scientific calculation.

## location.py
Methods to assign location to a given tweet or user profile.

## graph_ops.py
Methods for calculating several graph metrics.

## analyze.ipynb
To find early adopters and improve their profile information, find and analyze engagment, network structural, temporal, language, cascade and YouTube reaction of YouTube videos.

## additional_analysis.ipynb
Some additional analysis including cross-cutting exposure and sanity check for political leaning assignment.

## disparity_filtering.ipynb
To perform disparity filtering on early adopter's shared audience network to find its backbone.

## community_detection.ipynb
To find communities in shared audience network of early adopters (after disparity filtering).

## find_video_leaning_thresholds.ipynb
To calculate thr<sub>(L, C)</sub> and thr<sub>(C, R)</sub> based on the external source Recfluence.

## political_leaning_propagation.ipynb
To perform political leaning propagation on early adopters' shared audience network (after disparity filtering) to find political leaning scores of unknown users using the seed users' political leanings.

## prepare_online_offline_analysis_data.ipynb
To prepare online (video properties by leaning including leaning, engagement, network structural, temproal, language, cascade, and YouTube reaction measures and virality) and offline data relevant to topics for analysis.

## visualize_retweet_graph.ipynb
To visualize the retweets between the communities in shared audience network of early adopters obtained as a result of community detection.

## z_CSCW.ipynb
Methods to compare the measures of different topics as well as additional analysis for our submitted CSCW 2021 paper.
