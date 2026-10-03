from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open(r"C:\Users\agraw\OneDrive\Desktop\RAG_DEMO\Screenshot 2026-03-17 095910.png")  # use any image with text
text = pytesseract.image_to_string(img)

print(text)