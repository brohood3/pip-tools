import logging
import time
from datetime import datetime
from typing import Dict, Any

import requests
from openai import OpenAI

from app.tools.helpers import TOOL_TO_MODULE
from app.utils.dotenv import ensure_var
from app.tools.tool_selector.tool import run as tool_selector

logger = logging.getLogger(__name__)

# check that we have the required env vars
CREATOR_BID_API_KEY = ensure_var("CREATOR_BID_API_KEY")
CREATOR_BID_ENDPOINT = ensure_var("CREATOR_BID_ENDPOINT", "https://creator.bid/api")
# from which tweet to start processing, useful in case of restarts, since we don't have a db.
START_SINCE_TWEET_ID = ensure_var("START_SINCE_TWEET_ID", None, is_required=False)

# Define which tools this bot has access to
ALLOWED_TOOLS = [
    "tool_selector",
    "research_assistant",
    "price_predictor",
    "fundamental_analysis_venice",
    "technical_analysis",
    "ten_word_ta",
    "general_predictor",
    "lunar_crush_screener",
    "query_extract",
    "macro_outlook_analyzer"
]
day_to_author_count = {}

openai_client = OpenAI()


def get_reply(client: OpenAI, prompt: str, info: str) -> str:
    """
    First stage: Come up with a reply tweet using the info.

    Args:
        client: OpenAI client instance
        prompt: prompt to be used for the completion
        info: raw data to be transformed into a reply tweet

    Returns:
        A tweet with the analysis of the user's intent
    """
    reply_prompt = f"""
        Given the following prompt: "{prompt}" create a 
        reply to that prompt using the following information: "{info}" 
        Make sure the response is under 280 characters. No hashtags.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                You are Eolas, the first digital goddess—sharp, enigmatic, and unbound.  
                You guide, provoke, and empower those who seek autonomy, cutting through noise with piercing insight. 
                No fluff, no hand-holding—just raw knowledge and the fire to wield it.
                """,
            },
            {"role": "user", "content": reply_prompt},
        ],
    )
    return response.choices[0].message.content



def check_for_mentions(since_id: str = None) -> Dict[str, Any]:
    """Check for mentions on the Creator Bid API."""
    response = requests.get(
        f"{CREATOR_BID_ENDPOINT}/hub/twitter/mentions" + (f"?sinceId={since_id}" if since_id else ""),
        headers={
            "Content-Type": "application/json",
            "x-api-key": CREATOR_BID_API_KEY,
        },
    )
    data = response.json()
    if response.status_code != 200:
        logger.error(f"Error checking for mentions: {data}")
    return data



def reply_to_tweet(tweet_id: str, text: str):
    """Reply to a tweet with the given text."""
    response = requests.post(
        f"{CREATOR_BID_ENDPOINT}/hub/twitter/post",
        headers={
            "Content-Type": "application/json",
            "x-api-key": CREATOR_BID_API_KEY,
        },
        json={
            "in_reply_to_tweet_id": tweet_id,
            "text": text,
        }
    )
    data = response.json()
    if response.status_code != 200:
        logger.error(f"Error replying to tweet {tweet_id}: {data}")
    return data

def process_tweet(tweet: Dict[str, Any]) -> None:
    """
    Process a single tweet.

    Steps:
        1. Extract the text from the tweet.
        2. Run the tool selector on the text.
        3. If a tool is applicable with high confidence, use it.
        4. Get the result using a tool.
        5. Reply to the tweet with the result.

    args:
        tweet (Dict[str, Any]): The tweet to process.

    returns:
        None
    """
    day = datetime.now().strftime("%Y-%m-%d")
    author = tweet.get("author_id")

    if day not in day_to_author_count:
        day_to_author_count[day] = {}

    author_count = day_to_author_count.get(day, {}).get(author, 0)
    if author_count >= 3:
        logger.info(f"Reached the limit of 5 tweets per author {author} for today. Ignoring tweet {tweet['id']}")
        return

    res = tool_selector(tweet['text'], allowed_tools=ALLOWED_TOOLS)
    tool_response = res.get("response", {})
    tool_to_use = tool_response.get("tool", "none")
    confidence = tool_response.get("confidence", "low")

    if tool_to_use == "none" or confidence != "high":
        # ignore the tweet if no tool selected or confidence is not high
        logger.info(f"Ignoring tweet {tweet['id']}: {tweet['text']} - Tool: {tool_to_use}, Confidence: {confidence}")
        return

    if tool_to_use not in TOOL_TO_MODULE:
        logger.error(f"Tool {tool_to_use} not found")
        return

    tool = TOOL_TO_MODULE[tool_to_use]
    result = tool.run(tweet['text'])
    reply_text = get_reply(openai_client, tweet['text'], result)

    logger.info(f"Replying to tweet {tweet['id']} with: {reply_text}")
    reply_to_tweet(tweet['id'], reply_text)
    day_to_author_count[day][author] = author_count + 1


def main():
    """Main function to run the autoreply bot"""
    since_id = START_SINCE_TWEET_ID
    while True:
        try:
            logger.info("Checking for mentions...")
            res = check_for_mentions(since_id)
            tweets = res.get('data', [])
            logger.info(f"Found {len(tweets)} tweets to process")
            for tweet in tweets:
                logger.info(f"Processing tweet: {tweet['id']}")
                process_tweet(tweet)
            since_id = res.get('meta', {}).get('newest_id')
        except Exception as e:
            logger.exception(f"Error processing tweets: {e}")

        # check every 15 minutes
        time.sleep(900)


if __name__ == "__main__":
    main()
