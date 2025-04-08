from setuptools import setup, find_packages

setup(
    name="famma_runner",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "easyllm_kit",
        "datasets",
        "huggingface_hub",
    ],
    python_requires=">=3.8",
) 