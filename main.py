!pip install vaderSentiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


trump=pd.read_csv("/content/sample_data/hashtag_donaldtrump.csv")
biden=pd.read_csv("/content/sample_data/hashtag_joebiden.csv")




# Create a VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to apply VADER sentiment analysis to each tweet
def vader_pol(tweet):
    try:
        # Check if the tweet is not NaN or float
        if not pd.isna(tweet) and not isinstance(tweet, float):
            return analyzer.polarity_scores(tweet)['compound']
        else:
            return 0.0  # or any default value you prefer for NaN or non-string values
    except Exception as e:
        print(f"Error processing tweet: {tweet}. Error: {e}")
        return 0.0

# Apply VADER sentiment analysis to the 'tweet' column
trump["Vader_Polarity"] = trump["tweet"].apply(vader_pol)
biden["Vader_Polarity"] = biden["tweet"].apply(vader_pol)

# Categorize sentiments
trump["sentiment"] = pd.cut(trump["Vader_Polarity"], bins=[-1, -0.05, 0.05, 1], labels=["Negative", "Neutral", "Positive"])
biden["sentiment"] = pd.cut(biden["Vader_Polarity"], bins=[-1, -0.05, 0.05, 1], labels=["Negative", "Neutral", "Positive"])

# Display the dataframe with Vader sentiment scores and sentiments
print(trump[["tweet", "Vader_Polarity", "sentiment"]].head(100000))
print(biden[["tweet", "Vader_Polarity", "sentiment"]].head(100000))

# Generate summary statistics
summary_stats_trump= trump["Vader_Polarity"].describe()
print("\nSummary Statistics for VADER Polarity:\n", summary_stats_trump)

# Generate summary statistics
summary_stats_biden = biden["Vader_Polarity"].describe()
print("\nSummary Statistics for VADER Polarity:\n", summary_stats_biden)


count_trump=trump.groupby('sentiment').count()
count_biden=biden.groupby('sentiment').count()

print(count_trump)
print("\n")

print(count_biden)
print("\n")

name=["Trump","Biden"]
list_pos=[count_trump['Vader_Polarity'][2],count_biden['Vader_Polarity'][2]]
list_neg=[count_trump['Vader_Polarity'][0],count_biden['Vader_Polarity'][0]]
list_neut=[count_trump['Vader_Polarity'][1],count_biden['Vader_Polarity'][1]]


import plotly.graph_objects as go

fig=go.Figure(data=[
    go.Bar(name='+ve',x=name, y=list_pos),
    go.Bar(name='-ve', x=name, y=list_neg),
    go.Bar(name='Neut',x=name, y=list_neut)
])

fig.update_layout(barmode='group')
fig.show()



