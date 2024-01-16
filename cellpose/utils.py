import logging
import sys
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

from cellpose import models
import torch



def download_images(url, extract_to):
    '''
    From the github gist
    https://gist.github.com/hantoine/c4fc70b32c2d163f604a8dc2a050d5f6
    '''
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)
    
    
def setlogging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def check(gpu=True, device='mps'):
    model = models.Cellpose(gpu=gpu, device=device)
    return model
 
def checkmps():   
    d = torch.device('mps')    
    model = check(gpu=True, device=d)
    print(model.device)
    
    