import pandas as pd 
import requests
from bs4 import BeautifulSoup
import tqdm
import time
from korean_romanizer.pronouncer import Pronouncer


text = '이 세상엔 값진게 너무 많다.'
print(Pronouncer(text).pronounced)