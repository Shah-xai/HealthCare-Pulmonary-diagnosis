import setuptools
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
__version__="0.0.1"
setuptools.setup(
    name="CNN_Classifier",
    version=__version__,
    author="Shah",
    author_email="abedinn.shah@gmail.com",
    description="A CNN classifier for pulmonary diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shah-xai/HealthCare-Pulmonary-diagnosis",
)
package_dir={"":"src"}
setuptools.find_packages(where="src")
