# Standard
from importlib import metadata
from importlib.metadata import distribution

# Third Party
from pip._vendor.packaging.requirements import Requirement

PACKAGE_NAME = "vllm-detector-adapter"
VLLM_PACKAGE_NAME = "vllm"


def verify_vllm_compatibility(allow_prereleases=True):
    vllm_installed_version = metadata.version(VLLM_PACKAGE_NAME)

    vllm_required_version = ""

    vllm_adpt_dist = distribution(PACKAGE_NAME)

    # Extract vllm specified requirement from vllm-detector-adapter
    for package_info in vllm_adpt_dist.requires:
        requirement = Requirement(package_info)
        if requirement.name == VLLM_PACKAGE_NAME:
            vllm_required_version = requirement.specifier

    vllm_required_version.prereleases = allow_prereleases

    # Check if installed vllm version is compatible with PACKAGE_NAME requirements
    if vllm_installed_version in vllm_required_version:
        print("vLLM versions are compatible!")
    else:
        print(
            """
        Incompatible vLLM version installed. Please fix by either updating vLLM image or updating {} \n\n
        Installed vLLM version: {}
        {} required vLLM version range: {}
        """.format(
                PACKAGE_NAME,
                vllm_installed_version,
                PACKAGE_NAME,
                str(vllm_required_version),
            )
        )
        exit(1)
