# Lenny Leaderboard

This repository contains the source code for the Roboflow Lenny Leaderboard.

There are three parts to this repository:

1. `lennys_by_user.py`, which counts all Lennys sent at the time the script is run through the whole history of Slack, and;
2. `lenny.py`, which listens for Lennys given in Slack and adds them to a counter., and;
3. `render_lists.py`, which generates the Lenny Leaderboard HTML file.

The `lennys_by_user.py` script should only ever need to be run once when the tool is set up. It will take several minutes to run due to rate limiting.

To use this repository, first clone it and install the required dependencies:

```
git clone https://github.com/roboflow/lenny-leaderboard
pip install -r requirements.txt
```

Then, retrieve a Slack app and bot token and export them into your environment:

```
export SLACK_BOT_TOKEN="xoxp-..."
export SLACK_APP_TOKEN="xapp-1-..."
````

Then, run:

```
python3 lenny.py
```

A cron job should be set to run `render_lists.py` every hour. This will ensure the Lenny Leaderboard website is up to date.

## Contact

If you have questions about this project, please contact @James on Slack.