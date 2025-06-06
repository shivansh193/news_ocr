import os
import sys
import warnings
from .processor import NewsArticleOCR

# Apply global warning filter as in the original script
warnings.filterwarnings('ignore')

__version__ = "0.1.0"
__author__ = "Your Name" # Add your name
__email__ = "your.email@example.com" # Add your email


def process_news_article(image_path: str,
                         model_name: str = "microsoft/DialoGPT-small",
                         use_quantization: bool = True,
                         force_enhancement: bool = False,
                         display_results: bool = True) -> dict:
    """
    Processes a news article image: performs OCR, refines text with an LLM,
    and optionally displays the results.

    Args:
        image_path (str): Path to the input image file.
        model_name (str, optional): Name of the Hugging Face model to use for text refinement.
                                    Defaults to "microsoft/DialoGPT-small".
                                    Consider "google/flan-t5-small" for potentially better correction.
        use_quantization (bool, optional): Whether to use 4-bit quantization for the model.
                                           Defaults to True. Recommended for CausalLMs like DialoGPT.
                                           Set to False for models like FLAN-T5 if issues arise or
                                           if not using a CUDA-enabled GPU.
        force_enhancement (bool, optional): If True, image enhancement will always be applied.
                                            Defaults to False.
        display_results (bool, optional): If True, results (images and text) will be plotted
                                          using Matplotlib. Defaults to True.

    Returns:
        dict: A dictionary containing processing results, including keys like:
              "image_path", "initial_text", "final_cleaned_text", "initial_confidence",
              "initial_quality", "enhancement_used", "final_quality", etc.
              If an error occurs (e.g., image not found, Tesseract not found),
              an "error" key will be present in the dictionary with a description,
              and "final_cleaned_text" will be an empty string.

    Raises:
        Prints error messages to console for issues like TesseractNotFoundError or model loading errors.
        The function aims to catch exceptions and return an error dictionary instead of raising.
    """
    print("==============================================")
    print(" News Article OCR & LLM Enhancement Process ")
    print("==============================================")

    if not os.path.exists(image_path):
        error_msg = f"Error: Image not found at {image_path}"
        print(error_msg)
        return {
            "error": error_msg,
            "final_cleaned_text": ""
        }

    # Adjust quantization for specific models if needed, mirroring original script's intent
    actual_use_quantization = use_quantization
    if "flan-t5" in model_name.lower() and use_quantization:
        print(f"Warning: Quantization with BitsAndBytesConfig as specified might not be optimal "
              f"or directly supported for {model_name}. "
              "If issues arise or you are not using a CUDA GPU, consider setting use_quantization=False.")
        # For FLAN-T5, full 4-bit via BitsAndBytes might not be the standard path,
        # or it might not be beneficial without a GPU.
        # We'll proceed with user's choice but with a warning.

    processor = None
    try:
        processor = NewsArticleOCR(model_name=model_name, use_quantization=actual_use_quantization)
        results = processor.process_image(image_path, force_enhancement=force_enhancement)

        if display_results and "error" not in results:
            try:
                # Ensure matplotlib can be imported
                import matplotlib.pyplot
                processor.display_results(results)
            except ImportError:
                print("Matplotlib not found, cannot display results. Please install it (`pip install matplotlib`).")
            except Exception as e:
                print(f"Error displaying results: {e}. This might be a Matplotlib backend issue "
                      "if not running in an interactive environment like Jupyter.")

        # Print final text to console regardless of display_results, as per original flow
        if "error" not in results:
             print("\n--- Final Extracted and Cleaned Text ---")
             print(results.get('final_cleaned_text', "No text could be processed."))
        else:
            print(f"Processing failed. Error: {results.get('error')}")

        return results

    except ImportError as e:
        # Specific catch for missing critical dependencies, e.g. transformers, torch
        error_msg = f"A required library is missing: {e}. Please ensure all dependencies are installed."
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg, "final_cleaned_text": ""}
    except Exception as e:
        # Catch other exceptions like TesseractNotFoundError from processor, or model loading issues.
        error_msg = f"An error occurred during processing: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg, "final_cleaned_text": ""}
    finally:
        if processor:
            processor.cleanup_memory()
        print("\nProcessing finished.")

__all__ = ["process_news_article", "NewsArticleOCR"]