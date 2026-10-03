import base64

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage


def extract_images_from_pdf(pdf_path):
    """Return a list of (image_bytes, image_ext) tuples for every embedded
    image in the PDF. The extension is needed to build a correct mime type
    when sending the image to Gemini."""
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append((base_image["image"], base_image.get("ext", "png")))

    doc.close()
    return images


def analyze_image_with_gemini(image_bytes, image_ext, llm):
    prompt = """
You are an expert at understanding diagrams and graphs and tables.

Analyze the image and explain:
- What it represents
- Key components

Be clear and concise.
"""
    ext = "jpeg" if image_ext.lower() in ("jpg", "jpeg") else image_ext.lower()
    mime_type = f"image/{ext}"
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ]
    )

    response = llm.invoke([message])

    if isinstance(response.content, list):
        return " ".join(
            item.get("text", "") for item in response.content if isinstance(item, dict)
        )
    return response.content
