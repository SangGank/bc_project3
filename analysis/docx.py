from docx import Document

path = "../analysis/text_eng_spain_kor.docx"
# 워드 파일 열기
doc = Document(path)

# 단락 읽기
for para in doc.paragraphs:
    print(para.text)