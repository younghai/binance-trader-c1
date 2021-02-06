import logging
import requests
from config import CFG


URL = "<set url>"
TEST_URL = "<set url>"


class SlackHandler(logging.StreamHandler):
    def __init__(self):
        super(SlackHandler, self).__init__()
        self.url = TEST_URL if CFG.TEST_MODE is True else URL

    def emit(self, record):
        msg = self.format(record)
        self.send_message(msg)

    def send_message(self, text):
        if "[!] Error:" in text:
            text = "```" + text + " ```"
        else:
            text = ":sparkles: " + text

        message = {"text": text}

        requests.post(self.url, json=message)
