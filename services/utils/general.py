import boto3
import json
import time
from datetime import datetime
import pandas as pd


class SQS():
    sqs = boto3.client('sqs')

    def __init__(self, name):
        self.queue = self.sqs.create_queue(
            QueueName=name,
            Attributes={}
        )

        self.url = self.queue['QueueUrl']

    def send(self, message):
        res = self.sqs.send_message(
            QueueUrl=self.url,
            MessageBody=json.dumps(message),
            MessageAttributes={}
        )

        return res
    
    def recieve(self):
        res = self.sqs.receive_message(
            QueueUrl=self.url,
            MessageAttributeNames=['All']
        )

        try:
            message = json.loads(res['Messages'][0]['Body'])

            res = self.sqs.delete_message(
                QueueUrl=self.url,
                ReceiptHandle=res['Messages'][0]['ReceiptHandle']
            )

            return message
        except:
            return None
        
    def collect(self, n_items, max_wait=300):
        start = datetime.now()
        all_items = []
        while True:
            item = self.recieve()

            if item:
                all_items.append(item)

            if len(all_items) == n_items:
                self.self_destruct()
                break
            else:
                time.sleep(0.1)

            # force break after max wait
            if (datetime.now() - start).seconds > max_wait:
                self.self_destruct()
                break
        
        # sort items if order key is present
        try:
            all_items = json.loads(pd.DataFrame(all_items).sort_values(by='order').to_json(orient='records'))
        except:
            pass

        return all_items

    def self_destruct(self):
        _ = self.sqs.delete_queue(
            QueueUrl=self.url
        )



def get_user_id(username, wv_client=None):
  where_filter = {
    "path": ["username"],
    "operator": "Equal",
    "valueString": username
  }

  res = (
    wv_client.query
    .get("User", "_additional{ id }")
    .with_where(where_filter)
    .do()
  )

  return res['data']['Get']['User'][0]['_additional']['id']