from setuptools import setup, find_packages

setup(
    name='mini_gpt',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'langchain',
        'requests',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            # Add if needed
        ],
    },
)