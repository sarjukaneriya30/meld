import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContainerClient
from PIL import Image
import io
import base64
import numpy as np
import cv2
import ast


def getContainerClient(connect_str, container_name, credential = None):
    account_url = 'https://sharedaccount.blob.rack01.usmc5gsmartwarehouse.com/?sv=2019-02-02&ss=bqt&srt=sco&sp=rwdlacup&se=2022-11-18T05:28:27Z&st=2022-11-17T21:28:27Z&spr=https&sig=%2BVzVmfNwP0srU0d%2BKNW5iwTBPVQiqGKXPap4wxdBBBE%3D'
    container = ContainerClient(account_url, container_name = container_name, credential = credential, api_version = '2019-02-02')
    print(str(container.api_version))
    if container.exists():
        # Container foo exists. You can now use it.
        print(f'Container {container_name} already exists and is now ready to use.')

    else:
        # Container foo does not exist. You can now create it.
        container.create_container()
        print(f'Creating container {container_name} for use.')
    return container

def addDataFrameBlobs(df, filename, container):
    print(f'Uploading {filename} as blob to {container.container_name}...')
    # convert a dataframe to bytes and upload as a blob
    dfOut = df.to_csv(encoding='utf-8', index=False)
    container.upload_blob(filename, dfOut, overwrite=True, encoding='utf-8')
    return None

def addImageBlobs(container, photo_filename, photo_in_bytes):
    print(f'Uploading {photo_filename} as blob to {container.container_name}...')
    container.upload_blob(photo_filename, photo_in_bytes, overwrite=True)
    return None

def downloadSpecificDictBlob(blobname, container):
    blob = container.download_blob(blobname).readall()
    data = blob.decode()
    data = ast.literal_eval(data)
    return data

def downloadSpecificImageBlob(blobname, container_client):
    # Download blob into memory
    print(f'Downloading {blobname} to memory from {container_client.container_name}...')
    # downloads the blob as a bytes object
    blob = container_client.download_blob(blobname).readall()
    nparr = np.fromstring(blob, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

def listBlobs(container_client):
    # List the blobs in the container
    blob_list = container_client.list_blobs()
    return [blob.name for blob in blob_list]


'''
SUPPORTING CODE


#Code to upload templates to blob storage
file_path = '/Users/kritz/Documents/5GSWH/adv-5gswh-ocr-app/templates/json_files/'
filename = '1348-A_Old.json'
with open(file_path) as file:
    data = file.read()
    data = json.loads(data)
# add parameters
encoded_data = json.dumps(data).encode('utf-8')
blobStorage.addImageBlobs(templatejsonContainer, filename, encoded_data)

# Code to delete blobs
templatejsonContainer.delete_blob('1348-A_SHORT.json'
'''
