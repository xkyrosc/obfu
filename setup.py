from setuptools import setup, find_packages

setup(
    name="lua-discord-bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "discord.py>=2.3.0",
        "aiohttp>=3.9.0",
    ],
    python_requires=">=3.8",
)
