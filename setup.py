from setuptools import setup, find_packages

setup(
    name="chameleon-cache",
    version="1.0.0",
    description="A variance-adaptive cache replacement policy that beats TinyLFU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Chameleon Cache Authors",
    url="https://github.com/user/chameleon-cache",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Caching",
    ],
    keywords="cache, caching, lru, tinylfu, adaptive, algorithm",
)
