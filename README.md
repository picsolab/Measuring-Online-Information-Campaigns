# Evidence of Political Ideological Asymmetries from a Cross-Platform, Cross-Topic, Longitudinal Study

We release the code and data for the following paper.
If you use this dataset, or refer to its results, please cite:
> JooYoung Lee, Siqi Wu, Ali Mert Ertugrul, Yu-Ru Lin, and Lexing Xie. Whose Advantage? Measuring Attention Dynamics across YouTube and Twitter on Controversial Topics. *AAAI International Conference on Weblogs and Social Media (ICWSM)*, 2022. \[[paper](https://avalanchesiqi.github.io/files/icwsm2022xplatform.pdf)\]

## Data
We release tweeted video datasets for three controversial topics: Abortion (179 videos), Gun Control (268 videos), and BLM (777 videos).

### abo_videos.json
Each line contains cleaned video-level stats data for a YouTube video.

```json
{"video_id": "XUDtoDAGVE8",
 "title": "'Women's March' Crashed By Crowder... IN DRAG! (Featuring Wendy Davis)",
 "channel_title": "StevenCrowder",
 "channel_id": "UCIveFvW-ARp_B_RckhweNJw",
 "duration": 520,
 "political_label": "R",
 "start_date": "2017-01-24",
 "total_view120": 1795819,
 "daily_view120": [355460, 183161, 57041, 37288, 27380, 22026, 21257, ...],
 "avg_watch": 0.4946784519113252,
 "relative_engagement": 0.76,
 "num_like": 47324,
 "num_dislike": 831,
 "frac_like": 0.9827432250025958,
 "est_source": "GT",
 "est_prob": "0.8959362699353703",
 "total_tweet120": 6018,
 "daily_tweet120": [4106, 955, 149, 124, 72, 54, 19, ...],
 "exo": 86.76663468715331,
 "endo": 1.3272173220490222,
 "virality": 115.15818053268941
}
```

### Data Attribute Explanation
* video_id: video identifier on YouTube
* title: video title
* channel_title: title of YouTube channel
* channel_id: channel identifier on YouTube
* duration: video length in seconds
* political_label: ground-truth or predicted political leaning for this video
* start_date: the first day when the video was published
* total_view120: total view count in the first 120 days since the video's upload
* daily_view120: daily time series of view count in the first 120 days since the video's upload
* avg_watch: the average percentage all audience spend on this video
* relative_engagement: the percentile ranking of average watch percentage among videos of similar lengths
* num_like: the number of thumb-ups
* num_dislike: the number of thumb-downs (not visible on YouTube any more)
* frac_like: num_like / (num_like + num_dislike)
* est_source: the political leaning of this video is obtained from an external, expert-labeled dataset (i.e., "GT"") or classified by the label propagation algorithm (i.e., "inferred"")
* est_prob: the normalized score from label propagation
* total_tweet120: total tweet count in the first 120 days since the video's upload
* daily_tweet120: daily time series of tweet count in the first 120 days since the video's upload
* exo: exogenous promotion score. The number of views that one unit of promotion (i.e., a tweet) will lead to.
* endo: endogenous response score. The number of views that one unit of views (i.e., a view) will unfold.
* virality: virality score, defined as exo * endo. This metric quantifies the total return from one tweet.
