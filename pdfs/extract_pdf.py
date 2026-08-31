import pypdf

pdf_path = r"C:\Users\arkma\Documents\GitHub\SC-Project\pdfs\Baltic dry index forecast using financial market data Machine learning methods and SHAP explanations.pdf"
md_path = r"C:\Users\arkma\Documents\GitHub\SC-Project\pdfs\Baltic dry index forecast using financial market data Machine learning methods and SHAP explanations.md"

with open(pdf_path, "rb") as f:
    reader = pypdf.PdfReader(f)
    text = ""
    for i, page in enumerate(reader.pages):
        text += page.extract_text() + "\n\n"
        
with open(md_path, "w", encoding="utf-8") as f:
    f.write(text)
