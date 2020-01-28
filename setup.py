import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="segmentation_program", # Replace with your own username
    version="0.0.1",
    author="Luigi Masturzo & Sara Saponaro",
    author_email="author@example.com",
    description="A segmentation algorithm for mass lesions in mammography.",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://github.com/SaraSaponaro/segmentation_program",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
