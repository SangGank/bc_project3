import pandas as pd 
import requests
from bs4 import BeautifulSoup
import tqdm
import time
from korean_romanizer.pronouncer import Pronouncer

from hanspell import spell_checker
x = spell_checker.check('나태주 너를 먼저 생가칸다면 미투 나오 리 럽껟쬬')
print(x.as_dict()['checked'] ) # dict로 출력