from setuptools import find_packages, setup

# Read the content of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="KAFYTraj",
    packages=find_packages(include=["KAFY", "KAFY.*"]),
    package_data={"KAFY": ["transformersPlugin/**/*.py"]},  # Adjust the path as needed
    include_package_data=True,
    version="0.1.13",
    description="This library includes an extensible system for building various trajectory operations.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Change to "text/x-rst" for reStructuredText
    author="Youssef Hussein",
    install_requires=[
        "h3>=3.7.0",  # Specify the version range as needed
        # Add other dependencies here
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "kafy_parser=KAFY.commands_parser:parse_command",  # Optional: If you want to make it a CLI tool
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
