import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import hopsworks
from dotenv import load_dotenv, find_dotenv, dotenv_values


@st.cache_data(show_spinner=False)
def fetch_sentiment_data_from_hopsworks_fs() -> pd.DataFrame:
    """
    This function fetches the sentiment data for our tweets from the Hopsworks feature store

    Returns:
        pd.DataFrame: Predictions data
    """

    # Fetch variables for .env file
    print("Fetching API keys...")
    env_vars = dotenv_values(find_dotenv())
    hopsworks_api_key = env_vars["HOPSWORKS_API_KEY"]

    print("Fetching predictions data...")

    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    try:
        feature_view = fs.get_feature_view(
            name="nft_crypto_tweets_sentiments_view", version=1
        )
        df = feature_view.get_batch_data()
    except:
        iris_fg = fs.get_feature_group(
            name="nft_crypto_tweets_sentiment_predictions", version=1
        )
        query = iris_fg.select_all()
        feature_view = fs.create_feature_view(
            name="nft_crypto_tweets_sentiments_view",
            version=1,
            description="Read from Tweets sentiment dataset",
            labels=["dummy_target"],
            query=query,
        )
        df = feature_view.get_batch_data()
    return df


with st.spinner("Fetching sample prediction data from Hopsworks..."):
    df = fetch_sentiment_data_from_hopsworks_fs()
df["tweet_id"] = df["tweet_id"].astype("str")
df = df.sample(15)[
    [
        "tweet_text",
        "sentiment",
        "sentiment_score",
        "tweet_date",
        "tweet_id",
        "tweet_username",
    ]
]

st.header("Sample Prediction Data")
st.info("Double click on tweet_text cell to expand the text")
st.dataframe(df, hide_index=True, height=575)
