import pandas as pd
from pmaw import PushshiftAPI
import praw
import datetime as dt

api = PushshiftAPI()

reddit = praw.Reddit(
    client_id = client_id,
    client_secret = client_secret,
    user_agent = user_agent,
    username = username,
    password = password,
    check_for_async=False,
)



#Initialize dictionaries for storing comments

gunners_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }
chelseafc_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }
liverpoolfc_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }
mcfc_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }
reddevils_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }
coys_comments_dict = {
                "comment_body" : [],
                "comment_subreddit" : [],
                "comment_id" : [],
                "comment_parent_id" : []
                }                            

#Get the IDs of the daily discussion threads 
#from February 1st- February 5th 2023

start_epoch=int(dt.datetime(2023, 2, 1).timestamp())
end_epoch=int(dt.datetime(2023, 2, 6).timestamp())

ArsenalID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,
                            q = 'Daily Discussion',
                            subreddit='gunners',
                            filter=['id'],
                            limit=5))

ChelseaID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,
                            q = 'Daily Discussion',
                            subreddit='chelseafc',
                            filter=['id'],
                            limit=5))

LiverpoolID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,
                            q = 'Daily Discussion',
                            subreddit='liverpoolfc',
                            filter=['id'],
                            limit=5))

CityID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,         
                            q = 'Daily Discussion',
                            subreddit='mcfc',
                            filter=['id'],
                            limit=5))

UnitedID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,
                            q = 'Daily Discussion',
                            subreddit='reddevils',
                            filter=['id'],
                            limit=5))

SpursID = list(api.search_submissions(after=start_epoch,
                            before=end_epoch,
                            q = 'Daily Discussion',
                            subreddit='coys',
                            filter=['id'],
                            limit=5))


#Scrape the comments from the Reddit threads obtained above
#and store them in our dictionary. 
#Threads with a high comment count may need a search limit to prevent
#endless search

            
for day in ArsenalID:
      gunners_id = day['id']
      gunners_day =  reddit.submission(id=gunners_id)
      gunners_day.comments.replace_more(limit=50)
      for comment in gunners_day.comments.list():
          gunners_comments_dict["comment_body"].append(comment.body)
          gunners_comments_dict["comment_subreddit"].append(comment.subreddit)
          gunners_comments_dict["comment_id"].append(comment.id)
          gunners_comments_dict["comment_parent_id"].append(comment.parent_id)
                       
                      
for day in ChelseaID:
      chelseafc_id = day['id']
      chelseafc_day =  reddit.submission(id=chelseafc_id)
      chelseafc_day.comments.replace_more(limit=None)
      for comment in chelseafc_day.comments.list():
          chelseafc_comments_dict["comment_body"].append(comment.body)
          chelseafc_comments_dict["comment_subreddit"].append(comment.subreddit)
          chelseafc_comments_dict["comment_id"].append(comment.id)
          chelseafc_comments_dict["comment_parent_id"].append(comment.parent_id)

for day in LiverpoolID:
      liverpoolfc_id = day['id']
      liverpoolfc_day =  reddit.submission(id=liverpoolfc_id)
      liverpoolfc_day.comments.replace_more(limit=50)
      for comment in liverpoolfc_day.comments.list():
          liverpoolfc_comments_dict["comment_body"].append(comment.body)
          liverpoolfc_comments_dict["comment_subreddit"].append(comment.subreddit)
          liverpoolfc_comments_dict["comment_id"].append(comment.id)
          liverpoolfc_comments_dict["comment_parent_id"].append(comment.parent_id)            

for day in CityID:
      mcfc_id = day['id']
      mcfc_day =  reddit.submission(id=mcfc_id)
      mcfc_day.comments.replace_more(limit=None)
      for comment in mcfc_day.comments.list():
          mcfc_comments_dict["comment_body"].append(comment.body)
          mcfc_comments_dict["comment_subreddit"].append(comment.subreddit)
          mcfc_comments_dict["comment_id"].append(comment.id)
          mcfc_comments_dict["comment_parent_id"].append(comment.parent_id)

for day in UnitedID:
      reddevils_id = day['id']
      reddevils_day =  reddit.submission(id=reddevils_id)
      reddevils_day.comments.replace_more(limit=50)
      for comment in reddevils_day.comments.list():
          reddevils_comments_dict["comment_body"].append(comment.body)
          reddevils_comments_dict["comment_subreddit"].append(comment.subreddit)
          reddevils_comments_dict["comment_id"].append(comment.id)
          reddevils_comments_dict["comment_parent_id"].append(comment.parent_id)
          
for day in SpursID:
      coys_id = day['id']
      coys_day =  reddit.submission(id=coys_id)
      coys_day.comments.replace_more(limit=None)
      for comment in coys_day.comments.list():
          coys_comments_dict["comment_body"].append(comment.body)
          coys_comments_dict["comment_subreddit"].append(comment.subreddit)
          coys_comments_dict["comment_id"].append(comment.id)
          coys_comments_dict["comment_parent_id"].append(comment.parent_id)          

#Store comments in a dataframe

gunners = pd.DataFrame(gunners_comments_dict)
chelseafc = pd.DataFrame(chelseafc_comments_dict)
liverpoolfc = pd.DataFrame(liverpoolfc_comments_dict)               
mcfc = pd.DataFrame(mcfc_comments_dict)
reddevils = pd.DataFrame(reddevils_comments_dict)
coys = pd.DataFrame(coys_comments_dict)

#Mark each comment either as a reply to another comment or
#a completely original comment prefix t3 marks original comments

gunners["Reply"] = ""
chelseafc["Reply"] = ""
liverpoolfc["Reply"] = ""
mcfc["Reply"] = ""
reddevils["Reply"] = ""
coys["Reply"] = ""

for i in range(1,len(gunners)+1):
    if "t3" in str(gunners.iloc[i-1:i,3:]):
        gunners.iloc[i-1:i,4:] = "No"
    else: gunners.iloc[i-1:i,4:] = "Yes" 
    

for i in range(1,len(chelseafc)+1):
    if "t3" in str(chelseafc.iloc[i-1:i,3:]):
        chelseafc.iloc[i-1:i,4:] = "No"
    else: chelseafc.iloc[i-1:i,4:] = "Yes"
    
for i in range(1,len(liverpoolfc)+1):
    if "t3" in str(liverpoolfc.iloc[i-1:i,3:]):
        liverpoolfc.iloc[i-1:i,4:] = "No"
    else: liverpoolfc.iloc[i-1:i,4:] = "Yes"

for i in range(1,len(mcfc)+1):
    if "t3" in str(mcfc.iloc[i-1:i,3:]):
        mcfc.iloc[i-1:i,4:] = "No"
    else: mcfc.iloc[i-1:i,4:] = "Yes"    

for i in range(1,len(reddevils)+1):
    if "t3" in str(reddevils.iloc[i-1:i,3:]):
        reddevils.iloc[i-1:i,4:] = "No"
    else: reddevils.iloc[i-1:i,4:] = "Yes"
    
for i in range(1,len(coys)+1):
    if "t3" in str(coys.iloc[i-1:i,3:]):
        coys.iloc[i-1:i,4:] = "No"
    else: coys.iloc[i-1:i,4:] = "Yes"

#Rename columns for better readability

gunners = gunners[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]    
chelseafc = chelseafc[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]
liverpoolfc = liverpoolfc[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]
mcfc = mcfc[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]
reddevils = reddevils[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]
coys = coys[['comment_body','comment_subreddit','Reply','comment_id','comment_parent_id']]

gunners = gunners.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})
chelseafc = chelseafc.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})
liverpoolfc = liverpoolfc.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})
mcfc = mcfc.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})
reddevils = reddevils.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})
coys = coys.rename(columns={'comment_body': 'Comment', 'comment_subreddit': 'Club'})

for i in range(1,len(gunners)+1):
    gunners.iloc[i-1:i,1:2] = "Arsenal"
    
for i in range(1,len(chelseafc)+1):
   chelseafc.iloc[i-1:i,1:2] = "Chelsea"
    
for i in range(1,len(liverpoolfc)+1):
    liverpoolfc.iloc[i-1:i,1:2] = "Liverpool"

for i in range(1,len(mcfc)+1):
    mcfc.iloc[i-1:i,1:2] = "Manchester City"   

for i in range(1,len(reddevils)+1):
   reddevils.iloc[i-1:i,1:2] = "Manchester United"
    
for i in range(1,len(coys)+1):
   coys.iloc[i-1:i,1:2] = "Spurs"
   
#Save dataframes as CSV files   

gunners.to_csv('Arsenal.csv', index=False)
chelseafc.to_csv('Chelsea.csv', index=False)
liverpoolfc.to_csv('Liverpool.csv', index=False)
mcfc.to_csv('Manchester_City.csv', index=False)
reddevils.to_csv('Manchester_United.csv', index=False)
coys.to_csv('Spurs.csv', index=False)

#Combine datasets into one

bigsix = pd.concat([gunners,chelseafc,liverpoolfc,mcfc,reddevils,coys])

bigsix.to_csv('BigSix.csv', index=False)
