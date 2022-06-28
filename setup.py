from setuptools import setup, find_packages

name = "Python Image Laboratory"
version = "0.0.0"
release = "0.0"
setup(
    name=name,
    version=version,
    description="Python Image Processing/Analysis library",
    url="https://github.com/GassiusODude/py-imagelab",
    author="GassiusODude",
    license="MIT",
    packages=find_packages(),
    install_requires=['numpy', "opencv-python", "gradio", "dlib"
    ],
    dependency_links=[
    ],
    setup_requires=["nose>=1.3.7"],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs/source')
            }
        },
)
