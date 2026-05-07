from setuptools import setup, find_packages

setup(
    name="llm_dos_defense",
    version="0.1.0",
    description="Defense system against DoS attacks on Large Language Models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "transformers>=4.20.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
