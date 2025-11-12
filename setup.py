from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(path: str) -> list[str]:
    """Return non-empty, non-comment requirement lines."""
    reqs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="spider",
    version="0.1.0",
    description="Add your description here",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    packages=find_packages(include=["spider", "spider.*"]),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={"dev": ["ruff>=0.13.2"]},
    entry_points={"console_scripts": ["spider-mjwp=examples.run_mjwp:main"]},
)
