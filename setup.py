from setuptools import setup, find_packages

INSTALL_REQUIRES = []

setup(
    name="hommi", 
    author="Xiaomeng Xu",
    version="1.0.0",
    description="HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations",
    keywords="mobile manipulation, imitation learning",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=[],
    zip_safe=False
)
