from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name                = 'phaseaug',
    version             = '0.0.2',
    description         = 'PhaseAug: A Differentiable Augmentation for Speech Synthesis to Simulate One-to-Many Mapping',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author              = 'junjun3518',
    author_email        = 'junjun3518@gmail.com',
    url                 = 'https://github.com/mindslab-ai/phaseaug',
    install_requires    = ['torch', 'alias-free-torch'],
    packages            = ['phaseaug'],
    keywords            = ['torch','pytorch','augmentation', 'diffaugment', 'speech synthesis', 'vocoder'],
    python_requires     = '>=3',
    zip_safe            = False,
)
