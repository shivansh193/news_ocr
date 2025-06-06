# News Article OCR & LLM Enhancement Library (news_ocr_lib)

`news_ocr_lib` is a Python library designed to extract text from images of news articles using Tesseract OCR, and then refine that text using a Large Language Model (LLM) from the Hugging Face Transformers library. It includes image enhancement techniques to improve OCR accuracy and provides options for model quantization to save resources.

## Features

*   **Image Enhancement**: Applies various pre-processing techniques (denoising, thresholding, morphological operations) to improve image quality for OCR.
*   **OCR with Tesseract**: Extracts text and confidence scores.
*   **LLM-based Text Cleaning**: Uses a specified Hugging Face model (e.g., DialoGPT-small, FLAN-T5-small) to correct OCR errors, fix grammar, and improve readability.
*   **Quantization Support**: Leverages `bitsandbytes` for 4-bit model quantization (primarily for CUDA-enabled Linux systems) to reduce memory footprint.
*   **Result Visualization**: Option to display the original image, enhanced image (if used), and a summary of the extracted text using Matplotlib.
*   **Memory Management**: Includes utilities to clean up GPU memory after processing.

## Prerequisites

1.  **Python**: Python 3.7 or newer.
2.  **Tesseract OCR**: You **must** have Tesseract OCR installed on your system and available in your system's PATH.
    *   **Installation Instructions**: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   Ensure the `tesseract` command works in your terminal.
    *   You may also need to install language data (e.g., `tesseract-ocr-eng` for English).
3.  **CUDA (Optional, for GPU acceleration and `bitsandbytes` quantization)**:
    *   If you plan to use GPU acceleration and 4-bit quantization with `bitsandbytes`, you'll need a CUDA-compatible NVIDIA GPU and the appropriate NVIDIA drivers and CUDA toolkit installed. `bitsandbytes` primarily supports Linux for full 4-bit features. Windows support is partial or experimental.

## Installation

You can install the library using pip.

**Option 1: From PyPI (once published)**
```bash
pip install news_ocr_lib