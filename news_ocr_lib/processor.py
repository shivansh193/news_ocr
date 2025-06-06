import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import gc
import warnings
import os

# Suppress specific warnings if necessary, or manage globally in __init__.py
# warnings.filterwarnings('ignore') # Moved to __init__.py for package-level effect

class NewsArticleOCR:
    def __init__(self, model_name="microsoft/DialoGPT-small", use_quantization=True):

        self.model_name = model_name
        self.confidence_threshold = 65
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing NewsArticleOCR with model: {self.model_name} on device: {self.device}")

        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model_kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16 # Use float16 on GPU if not quantizing

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ Model and tokenizer loaded successfully.")

    def enhance_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )

        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        enhanced_img = Image.fromarray(processed)
        enhanced_img = enhanced_img.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = enhancer.enhance(1.5)

        return enhanced_img

    def extract_text_with_confidence(self, image: Image.Image) -> Tuple[str, float]:
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng')
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract is not installed or not in your PATH.")
            print("Please install Tesseract OCR: https://tesseract-ocr.github.io/tessdoc/Installation.html")
            raise

        text_parts = []
        confidences = []

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > -1 and data['text'][i].strip():
                text_parts.append(data['text'][i])
                if int(data['conf'][i]) > 0:
                    confidences.append(int(data['conf'][i]))

        extracted_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return extracted_text, avg_confidence

    def _basic_text_clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[|]{2,}', ' ', text)
        text = re.sub(r'[-_]{3,}', ' ', text)
        return text

    def _query_llm(self, text_to_clean: str) -> str:
        if not text_to_clean.strip():
            return ""

        if "flan-t5" in self.model_name.lower():
            prompt = (
                f"Correct OCR errors in the following news article excerpt. "
                f"Focus on fixing misspellings, merging broken words, removing extraneous "
                f"characters or OCR artifacts (like excessive hyphens or vertical bars), "
                f"and ensuring grammatical correctness. Preserve the original meaning and "
                f"sentence structure as much as possible. Do not add new information or summarize.\n\n"
                f"Text to correct:\n'''{text_to_clean}'''\n\n"
                f"Corrected text:"
            )
        else:
             prompt = (
                f"The following text was extracted from a news article using OCR and contains errors. "
                f"Please correct these errors. This includes fixing misspellings, joining fragmented words, "
                f"and removing nonsensical characters or repeated symbols (e.g., '---', '|||'). "
                f"Ensure the corrected text reads naturally and maintains the original meaning. \n\n"
                f"Original Text:\n'''{text_to_clean}'''\n\n"
                f"Corrected Text:\n"
            )

        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = inputs.to(self.device)

            output_sequence_length = min(int(len(inputs[0]) * 1.5) + 50, self.tokenizer.model_max_length or 1024)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=output_sequence_length,
                    temperature=0.2,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    early_stopping=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Corrected Text:" in response:
                cleaned_text = response.split("Corrected Text:")[-1].strip()
            elif prompt.strip().endswith("Corrected Text:"):
                 cleaned_text = response.replace(prompt.replace("Corrected Text:", ""),"").strip()
            elif "'''" in response:
                 cleaned_text = response.split("'''")[-1].strip()
            else:
                cleaned_text = response.replace(prompt, "").strip()

            if len(cleaned_text) < len(text_to_clean) * 0.7 or cleaned_text.strip() == text_to_clean.strip() or not cleaned_text.strip():
                print("LLM output quality low or no change, using basic cleaning for this segment.")
                return self._basic_text_clean(text_to_clean)

            return self._basic_text_clean(cleaned_text)

        except Exception as e:
            print(f"LLM processing error: {e}. Falling back to basic cleaning.")
            return self._basic_text_clean(text_to_clean)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        total_chars = len(text)
        if total_chars == 0:
            return {"quality_score": 0, "needs_enhancement": True, "issues": ["No text extracted"], "word_count": 0, "avg_sentence_length": 0}

        issues = []
        quality_score = 100
        words = text.split()
        word_count = len(words)

        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()\-%]', text)) / total_chars
        if special_char_ratio > 0.08:
            issues.append(f"High special char ratio ({special_char_ratio:.2f})")
            quality_score -= 20

        if word_count > 0:
            avg_word_len = sum(len(w) for w in words) / word_count
            if avg_word_len < 3.5:
                issues.append(f"Very short average word length ({avg_word_len:.1f})")
                quality_score -= 20

            short_words = [w for w in words if len(w) <= 2 and w.isalpha() and w.lower() not in ['a', 'i', 'o', 'an', 'is', 'it', 'in', 'to', 'of', 'on', 'or', 'at', 'by', 'my', 'me', 'he', 'we', 'us', 'so', 'no', 'go', 'do', 'up', 'as', 'be']]
            if len(short_words) / word_count > 0.25:
                issues.append(f"Many potentially fragmented words ({len(short_words)} of {word_count})")
                quality_score -= 25

        if re.search(r'[|_-]{2,}', text):
            issues.append("Contains multiple consecutive separators like '---' or '||'")
            quality_score -= 15

        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 1]
        avg_sentence_length = np.mean([len(s.split()) for s in valid_sentences]) if valid_sentences else 0
        if valid_sentences and avg_sentence_length < 4:
            issues.append(f"Short avg sentence length ({avg_sentence_length:.1f} words)")
            quality_score -= 15

        needs_enhancement = quality_score < 75 or len(issues) > 0

        return {
            "quality_score": max(0, quality_score),
            "needs_enhancement": needs_enhancement,
            "issues": issues,
            "word_count": word_count,
            "avg_sentence_length": avg_sentence_length
        }

    def process_image(self, image_path: str, force_enhancement: bool = False) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            # This check is also done in the main API function, but good to have defensively
            error_msg = f"Error: Image not found at {image_path}"
            print(error_msg)
            return {"error": error_msg, "final_cleaned_text": ""}

        print(f"\n--- Processing: {os.path.basename(image_path)} ---")

        original_img = Image.open(image_path)
        extracted_text, initial_confidence = self.extract_text_with_confidence(original_img)
        initial_quality = self._assess_text_quality(extracted_text)

        print(f"Initial OCR: Confidence {initial_confidence:.1f}%, Quality Score {initial_quality['quality_score']}/100")
        if initial_quality['issues']:
            print(f"Initial issues: {', '.join(initial_quality['issues'])}")

        current_text = extracted_text
        current_confidence = initial_confidence
        enhancement_used = False
        enhanced_text_content = None

        if force_enhancement or initial_confidence < self.confidence_threshold or initial_quality['needs_enhancement']:
            print("Applying image enhancement...")
            enhancement_used = True
            try:
                enhanced_img = self.enhance_image(image_path)
                enhanced_text, enhanced_confidence = self.extract_text_with_confidence(enhanced_img)
                print(f"Enhanced OCR: Confidence {enhanced_confidence:.1f}%")

                if enhanced_confidence > initial_confidence + 5 or \
                   (abs(enhanced_confidence - initial_confidence) <= 5 and len(enhanced_text.split()) > len(extracted_text.split()) * 1.1):
                    print("Using text from enhanced image.")
                    current_text = enhanced_text
                    current_confidence = enhanced_confidence
                    enhanced_text_content = enhanced_text
                else:
                    print("Enhancement did not significantly improve OCR. Using initial OCR text.")
                    enhanced_text_content = enhanced_text
            except Exception as e:
                print(f"Error during image enhancement: {e}. Using initial OCR text.")
        else:
            print("Skipping image enhancement based on initial quality.")

        cleaned_ocr_text = self._basic_text_clean(current_text)
        print("Applying LLM text refinement...")
        llm_refined_text = self._query_llm(cleaned_ocr_text)
        final_quality = self._assess_text_quality(llm_refined_text)
        print(f"Final Text Quality Score: {final_quality['quality_score']}/100")

        return {
            "image_path": image_path,
            "initial_text": extracted_text,
            "initial_confidence": initial_confidence,
            "initial_quality": initial_quality,
            "enhancement_used": enhancement_used,
            "enhanced_text_from_ocr": enhanced_text_content,
            "text_before_llm": cleaned_ocr_text,
            "final_cleaned_text": llm_refined_text,
            "final_quality": final_quality
        }

    def display_results(self, results: Dict[str, Any]):
        if "error" in results or not results.get("image_path"):
            print(f"Cannot display results due to error: {results.get('error', 'Unknown error')}")
            return

        image_path = results["image_path"]
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        try:
            img = Image.open(image_path)
            plt.imshow(img)
        except FileNotFoundError:
            plt.text(0.5, 0.5, "Image not found", ha='center', va='center')
        plt.title(f"Original: {os.path.basename(image_path)}\nInitial OCR Conf: {results.get('initial_confidence', 0):.1f}%")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        if results.get('enhancement_used'):
            try:
                enhanced_display_img = self.enhance_image(image_path)
                plt.imshow(enhanced_display_img, cmap='gray')
                plt.title(f"Enhanced Image (if used for OCR)")
            except Exception as e:
                plt.text(0.5, 0.5, f"Error displaying\nenhanced image:\n{e}", ha='center', va='center', fontsize=9)
        else:
            plt.text(0.5, 0.5, "Enhancement not used", ha='center', va='center')
        plt.axis('off')

        plt.subplot(2, 1, 2)
        info_text = (
            f"Initial Quality: {results.get('initial_quality', {}).get('quality_score', 'N/A')}/100 "
            f"| Enhancement Used: {results.get('enhancement_used', False)}\n"
            f"Final Quality: {results.get('final_quality', {}).get('quality_score', 'N/A')}/100 "
            f"| Final Word Count: {results.get('final_quality', {}).get('word_count', 'N/A')}\n"
            f"Initial Issues: {', '.join(results.get('initial_quality', {}).get('issues', ['None']))}\n"
        )
        plt.text(0.01, 0.98, "Processing Summary:", fontsize=12, fontweight='bold', va='top', transform=plt.gca().transAxes)
        plt.text(0.01, 0.9, info_text, fontsize=9, va='top', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6))
        plt.text(0.01, 0.75, "Final Cleaned Text:", fontsize=12, fontweight='bold', va='top', transform=plt.gca().transAxes)
        cleaned_text_display = results.get('final_cleaned_text', "No text processed.")
        if len(cleaned_text_display) > 800:
            cleaned_text_display = cleaned_text_display[:800] + "\n... (text truncated for display)"
        plt.text(0.01, 0.7, cleaned_text_display, fontsize=10, va='top', wrap=True, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        plt.axis('off')

        plt.tight_layout(pad=2.0)
        plt.show()

    def cleanup_memory(self):
        print("Cleaning up model and tokenizer from memory...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ Memory cleanup attempted.")