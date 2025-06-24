from news_ocr_lib import NewsArticleOCR

processor = NewsArticleOCR(model_name="microsoft/DialoGPT-small", use_quantization=True)



image_file = "path to your image here"
if not image_file:
    image_file = input("Please enter the path to your image: ")
processing_results = processor.process_image(image_file, force_enhancement=False)

if "error" not in processing_results:
    # Display results (optional)
    processor.display_results(processing_results)

    print("\n--- Final Cleaned Text ---")
    print(processing_results["final_cleaned_text"])
else:
    print(f"Error: {processing_results['error']}")

# Important: Clean up model from memory when done
processor.cleanup_memory()