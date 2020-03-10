import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rmvsnet",
    version="0.0.1",
    author="Jae Yong Lee",
    author_email="lee896@illinois.edu",
    description="R-MVSNet: Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leejaeyong7/rmvsnet-pytorch",
    packages=setuptools.find_packages(),
    package_data={
      'weights': ['rmvsnet/RMVSNET-pretrained.pth']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
