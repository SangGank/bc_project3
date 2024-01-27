from docx import Document
import pandas as pd

path = "/data/ephemeral/bc_project3/analysis/text_japan_kor.docx"
# 워드 파일 열기
doc = Document(path)

li=[]
# 단락 읽기
for para in doc.paragraphs:
    # print(para.text)
    li.append(para.text)
df = pd.read_csv('./data/train_p2g.csv')
df2 =df.copy()
df2.text=li
print(df2.head())
print(df.head())
df2.to_csv('./data/back_translate_jap.csv',index=False)

# print(li[:5])