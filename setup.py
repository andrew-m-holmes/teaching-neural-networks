from setuptools import setup, find_packages

with open("./requirements.txt", mode="r") as file:
    requirements = [line.strip() for line in file.readlines()]

setup(
    name="teaching neural networks",
    version="0.1.0",
    author="Andrew Holmes",
    author_email="andrewholmes011002@gmail.com",
    url="https://github.com/andrew-m-holmes/teaching-neural-networks",
    python_requires=">=3.9",
    packages=find_packages(exclude=["*venv*"]),
    install_requires=requirements,
)
