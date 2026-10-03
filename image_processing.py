from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.

    Steps:
    1. Convert to grayscale
    2. Increase contrast
    3. Apply sharpening
    """

    # 1. Convert to grayscale
    img = img.convert("L")

    # 2. Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # 3. Sharpen image
    img = img.filter(ImageFilter.SHARPEN)

    return img


def advanced_preprocess_image(img: Image.Image) -> Image.Image:
    """
    More aggressive preprocessing (optional).
    Use for very noisy/scanned images.
    """
    # 4. Noise removal
    img = img.filter(ImageFilter.MedianFilter())

    # Grayscale
    img = img.convert("L")

    # Increase contrast more
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)

    # Slight blur (reduce noise)
    img = img.filter(ImageFilter.MedianFilter())

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # 6. Thresholding (binarization)
    img = img.point(lambda x: 0 if x < 140 else 255)

    return img