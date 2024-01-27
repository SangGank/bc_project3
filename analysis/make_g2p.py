from g2pk import G2p
import pandas as pd

g2p = G2p()

ai = pd.read_csv('./data/train_ai_limit.csv')
ai['type']=0

ai_per = ai.copy()
ai_des = ai.copy()
ai_per.type=1
ai_des.type=2
ai.to_csv('./data/train_ai_type.csv', index =False)

ai_per.text= ai_per.text.apply(lambda x: g2p(x))
ai_per.to_csv('./data/train_ai_per.csv', index =False)

ai_des.text= ai_des.text.apply(lambda x: g2p(x,descriptive=True))
ai_des.to_csv('./data/train_ai_des.csv',index =False)
