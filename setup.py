import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mob-face-recognition",  # Replace with your own username
    version="0.0.1",
    install_requires=[
        "opencv-python~=4.2",
        "PyYAML~=6.0",
        "torch~=1.10",
        "faceboxes_pytorch @ git+https://github.com/Liquid-dev/FaceBoxes.PyTorch.git@0.1.3",
    ],
    author="taroogura",
    author_email="sard505@gmail.com",
    description="Mobile Face Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taroogura/mob-face-recognition.git",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
