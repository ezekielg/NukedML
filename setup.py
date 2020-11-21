from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='NukedML',
    version='0.1',
    author='Ezekiel Gomez',
    author_email='ezekielg@uw.edu',
    license='MS-PL',
    url='https://github.com/ezekielg/NukedML/',
    description='Hmmm, super mysterious.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Topic :: Data Engineering',
        'Topic :: Data Science',
        'Topic :: Machine Learning',
        'Intended Audience :: Developers'
    ],
    python_requires='>=3.7'
)