import pandas as pd 
import requests
from bs4 import BeautifulSoup
import tqdm
import time



def findRealName(url):
    time.sleep(5)
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'html.parser')
    if 'sports' in url:
        tag= '#content > div > div.content > div > div.news_headline > h4'
    else: 
        tag= '#title_area > span'
    title = soup.select(tag)
    title= str(title)
    
    return title



test_data = pd.read_csv('./data/test.csv')
test_data2 = test_data.copy()
test_data2['real'] = test_data2.url.apply(lambda x: findRealName(x))

test_data2.to_csv('./data/test_Name.csv')
