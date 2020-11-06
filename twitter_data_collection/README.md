## partition user_ids for data collection w.r.t api keys
Partition the user_ids before collection of their followers/friends in parallel using the following command.

`python3 partition_userids.py --campaign gun`

## collect followers/friends ids 
Run "collect_user_followers_friends.py" for each chunk to perform parallel collection as follows:

`nohup python3 collect_user_followers_friends.py --campaign gun --type followers --chunk_no 0 &`
`nohup python3 collect_user_followers_friends.py --campaign gun --type followers --chunk_no 1 &`
`nohup python3 collect_user_followers_friends.py --campaign gun --type followers --chunk_no 2 &`

## create graph edges 
Calculating graph weights is expensive. The following command will make it in multi-process way.

`nohup python3 multiprcs.py &`