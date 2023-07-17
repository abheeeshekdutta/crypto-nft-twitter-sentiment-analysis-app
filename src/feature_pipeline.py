import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import pandas as pd
import json
from dotenv import load_dotenv, find_dotenv, dotenv_values
import hopsworks
import numpy as np


def get_response(url: str, querystring: dict, headers: dict) -> dict:
    """
    Return a Python dictionary object of the JSON response received from the API.
    Args:
        url (str): URL of the API
        querystring (dict): Parameters for API request
        headers (dict): Headers for API request

    Returns:
        dict: Dictionary object of the response
    """
    response = requests.get(url, headers=headers, params=querystring)
    dict_object = response.json()
    return dict_object


def transform_json_to_dataframe(json_obj: dict) -> pd.DataFrame:
    """
    This function returns a Pandas Dataframe with the following columns
    extracted from the JSON object:
    1. Tweet's ID
    2. Tweet's date
    3. Tweet's full text
    4. Screen name of the user who published the tweet

    For sentiment analysis, the tweet full text is key.

    Args:
        json_obj (dict): Dictionary of all responses received from the API.

    Returns:
        pd.DataFrame: Results dataframe containing the four mentioned columns.
    """
    tweet_ids = []
    tweet_text = []
    tweet_date = []
    tweet_username = []

    for record in json_obj["statuses"]:
        tweet_ids.append(record["id"])
        tweet_date.append(
            datetime.strptime(record["created_at"], "%a %b %d %H:%M:%S %z %Y")
        )
        tweet_text.append(record["full_text"])
        tweet_username.append(record["user"]["screen_name"])

    tweets_df = pd.DataFrame(
        {
            "tweet_id": tweet_ids,
            "tweet_date": tweet_date,
            "tweet_text": tweet_text,
            "tweet_username": tweet_username,
        }
    )

    tweets_df["dummy_target"] = np.random.randint(0, 100, size=len(tweets_df))

    print(f"Total tweets fetched: {tweets_df.shape[0]}")

    return tweets_df


def push_to_hopsworks_fs(tweets_df: pd.DataFrame):
    """
    This function pushes raw tweets data to the Hopsworks feature group

    Args:
        tweets_df (pd.Dataframe): Tweets dataframe that needs to be pushed to Hopsworks feature store
    """
    hopsworks_api_key = env_vars["HOPSWORKS_API_KEY"]
    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()
    tweets_raw_fg = fs.get_or_create_feature_group(
        name="nft_crypto_tweets_raw",
        version=2,
        primary_key=["tweet_id"],
        description="Feature group for storing raw tweets data",
    )
    tweets_raw_fg.insert(features=tweets_df, overwrite=False, operation="upsert")


if __name__ == "__main__":
    # Fetch variables for .env file
    print("Fetching API keys...")
    env_vars = dotenv_values(find_dotenv())

    # Set variables
    url = env_vars["url"]
    X_RapidAPI_Key = env_vars["X_RapidAPI_Key"]
    X_RapidAPI_Host = env_vars["X_RapidAPI_Host"]

    headers = {"X-RapidAPI-Key": X_RapidAPI_Key, "X-RapidAPI-Host": X_RapidAPI_Host}

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() + relativedelta(days=-1)).strftime("%Y-%m-%d")

    querystring = {
        "q": f"#nft #crypto nft crypto -filter:retweets since:{yesterday}",
        "count": "100",
        "result_type": "mixed",
        "lang": "en",
    }

    # Fetch data from API
    print("Fetching data from API...")
    json_obj = get_response(url, querystring=querystring, headers=headers)

    # Transform API data into Pandas Dataframe
    print("Creating tweets dataframe...")
    tweets_df = transform_json_to_dataframe(json_obj)

    # Push data to Hopsworks feature group
    print("Pushing data to hopsworks feature group...")
    push_to_hopsworks_fs(tweets_df)

    print("Saving data locally as a parquet file...")
    tweets_df.to_parquet("../data/tweets_df.parquet")
