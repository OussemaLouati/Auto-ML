import pandas as pd
import logging
import os
from google.cloud import storage

from .loader import Loader

class GoogleCloudObjectLoader(Loader):
    def __init__(self, key_path, bucket_name):
        # initialize client and fetch bucket
        try:
            storage_client = storage.Client.from_service_account_json(key_path)
            self.bucket = storage_client.bucket(bucket_name)
        except Exception as e:
            logging.error(str(e))
            raise SystemExit

    def load(self, resource_name):
        blob = self.bucket.blob(resource_name)
        blob.download_to_filename(resource_name)
        dataset = pd.read_csv(resource_name)
        os.remove(resource_name)
        return dataset

    def save(self, dataset, resource_name):
        dataset.to_csv(resource_name, index=False)
        blob = self.bucket.blob(resource_name)
        blob.upload_from_filename(resource_name)    
        os.remove(resource_name)

