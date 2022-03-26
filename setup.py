'''
Function:
    setup the pytoydl
Author:
    Charles
微信公众号:
    Charles的皮卡丘
GitHub:
    https://github.com/CharlesPikachu
'''
import pytoydl
from setuptools import setup, find_packages


'''readme'''
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


'''setup'''
setup(
    name=pytoydl.__title__,
    version=pytoydl.__version__,
    description=pytoydl.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent'
    ],
    author=pytoydl.__author__,
    url=pytoydl.__url__,
    author_email=pytoydl.__email__,
    license=pytoydl.__license__,
    include_package_data=True,
    package_data=package_data,
    install_requires=list(open('requirements.txt', 'r').readlines()),
    zip_safe=True,
    packages=find_packages(),
)