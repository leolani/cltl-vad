from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.vad',
    version=version,
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['cltl.*', 'cltl_service.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-vad",
    license='MIT License',
    author='CLTL',
    author_email='t.baier@vu.nl',
    description='VAD for Leolani',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=["numpy>=1.21.2"],
    extras_require={
        "impl": [
            "soundfile>=0.10.3.post1",
            "webrtcvad>=2.0.10",
            "parameterized"
        ],
        "service": [
            "cltl.backend",
            "cltl.combot",
            "emissor",
            "requests"
        ]
    }
)
