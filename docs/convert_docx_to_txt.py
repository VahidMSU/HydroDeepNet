from docx import Document
import os
from docx.opc.exceptions import PackageNotFoundError




def docx_to_text(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])


path = '/data/SWATGenXApp/codes/docs/old_docs/recharge paper.docx'
base_path = '/data/SWATGenXApp/codes/docs/old_docs/'
for file in os.listdir(base_path):
    if file.endswith('.docx'):
        path = os.path.join(base_path, file)
        try:
            text = docx_to_text(path)
            with open(os.path.join(base_path, file.replace('.docx', '.txt')), 'w') as f:
                f.write(text)
            print(f"Successfully converted: {file}")
        except PackageNotFoundError:
            print(f"Error: Could not open file as a valid docx package: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")