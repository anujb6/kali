from nasaapi import Client
from PIL import Image
import urllib.request
nasa = Client("")

def Apod_Nasa():
    data = nasa.apod()
    with urllib.request.urlopen(data['url']) as url:
        img = Image.open(url)
        img.show()
    return data['title'] + " " + data['explanation']


