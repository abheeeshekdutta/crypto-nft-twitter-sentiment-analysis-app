import streamlit as st
from transformers import pipeline
import pandas as pd
import torch
import hopsworks
from dotenv import load_dotenv, find_dotenv, dotenv_values

st.write("testing")


@st.cache_data
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


x = st.slider("x")  # 👈 this is a widget
st.write(x, "squared is", x * x)

df = fetch_sentiment_data_from_hopsworks_fs()
st.dataframe(df.sample(5), hide_index=True)
