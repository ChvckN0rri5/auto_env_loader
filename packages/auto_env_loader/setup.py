from setuptools import setup, find_packages

setup(
    name='auto_env_loader',
    version='0.1.0',
    packages=find_packages(),
    description='A simple package to automatically load .env files into environment variables.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ChvckN0rri5', 
    url='https://github.com/ChvckN0rri5/auto_env_loader', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    # No external dependencies needed for this simple version
    # If you were using python-dotenv, you'd add it here:
    # install_requires=[
    #     'python-dotenv>=0.15.0',
    # ],
)
