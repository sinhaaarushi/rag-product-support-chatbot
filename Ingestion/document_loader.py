import fitz
import os

def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)

            pdf = fitz.open(file_path)
            text = ""

            for page in pdf:
                text += page.get_text()

            documents.append({
                "file_name": file,
                "text": text
            })

    return documents