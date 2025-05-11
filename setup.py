from setuptools import find_packages, setup

setup(
    name="POPGym Arcade",
    version="0.0.1",
    author="Wang Zekang, He Zhe, Steven Morad",
    author_email="",
    description="POMDP Arcade Environments on the GPU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        # gymnax does not work with newest version of jax
        # use stable-gymnax alternative
        "gymnax @ git+ssh://git@github.com/smorad/stable-gymnax.git",
        "dm_pix",
        "jaxtyping",
    ],
    extras_require = {
        "baselines": [
            "optax",
            "equinox",
            "distreqx",
            "wandb",
            "beartype",
            "jaxtyping",
            "imageio"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)