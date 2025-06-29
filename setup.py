import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name='mcp_style_doc_parse_chatbot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'faiss-cpu',
        'ollama',
        'pymupdf',
        'python-docx',
    ],
    entry_points={
        'console_scripts': [
            'mcp-style-doc-parse-chatbot=mcp_style_doc_parse_chatbot.chatbot:main',
        ],
    },
    author='Your Name',
    description='CLI to query PDF and DOCX with local LLM and FAISS retrieval',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.7',
)