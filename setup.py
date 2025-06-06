from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='news_ocr_lib',  # Package name on PyPI
    version='0.1.0',     # Initial version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='OCR and LLM-based text cleaning for news articles.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/news_ocr_package',  # Replace with your GitHub repo URL (optional)
    project_urls={ # Optional
        'Bug Tracker': 'https://github.com/yourusername/news_ocr_package/issues',
    },
    license='MIT',  # Choose a license (e.g., MIT, Apache 2.0) and add a LICENSE file
    packages=find_packages(where='.'), # find_packages() will find 'news_ocr_lib'
    # package_dir={'': '.'}, # Not needed if packages are directly under root with find_packages()
    install_requires=[
        'pytesseract>=0.3.8',
        'opencv-python>=4.5',
        'Pillow>=9.0',
        'matplotlib>=3.5', # For display_results
        'numpy>=1.21',
        'transformers>=4.24', # Check for a version compatible with your model choices
        'torch>=1.12',        # Or a specific version for CUDA/CPU
        'accelerate>=0.15',
        'bitsandbytes>=0.35', # For 4-bit quantization, primarily on Linux with CUDA
    ],
    classifiers=[
        'Development Status :: 3 - Alpha', # Or '4 - Beta', '5 - Production/Stable'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing :: Optical Character Recognition (OCR)',
        'License :: OSI Approved :: MIT License',  # Match your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent', # Base Python code
        # Add note about bitsandbytes usually needing Linux for full features
    ],
    python_requires='>=3.7', # Specify your Python version compatibility
    keywords='ocr, llm, text extraction, news article, image processing, transformers', # Optional
)