#Import library
from PyPDF2 import PdfReader 

#Load PDF file
pdf_path="genai-principles.pdf"
reader=PdfReader(pdf_path)
total_pages=len(reader.pages)

#Print the total number of pages
print(f"\n Total Pages:{total_pages}")

#Extract text from pdf and count words in pdf
all_text=""
for i,page in enumerate(reader.pages):
    text=page.extract_text()
    print(f"\n---Page{i+1}---\n{text}")
    all_text+=text+" "

word_count=len(all_text.split())
print(f"\nTotal words:{word_count}")

#save the extracted TEXT to a .txt File
with open("Extracted_output.txt","w",encoding="utf-8") as f:
    f.write(all_text)
print("\n Text saved to extracted_output.txt")



