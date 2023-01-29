import pandas as pd
from pmaw import PushshiftAPI
import praw
import datetime as dt

api = PushshiftAPI()

start_epoch=int(dt.datetime(2023, 1, 1).timestamp())

ArsenalID = list(api.search_submissions(after=start_epoch,
                            q = 'Daily Discussion',
                            subreddit='gunners',
                            filter=['id'],
                            limit=10))




reddit = praw.Reddit(
    client_id = client_id,
    client_secret = client_secret,
    user_agent=user_agent,
)

submission = reddit.submission("10mfky2")

submission.comments.replace_more(limit=0)
for top_level_comment in submission.comments:
    print(top_level_comment.body)