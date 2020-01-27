'''
import os
import logging


# Basic local environment.
#
BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _join(*args) -> str:
    """Path concatenation relatove to the bas package folder (avoids some typing.)
    """
    return os.path.join(BASE_FOLDER, *args)


ROOT_FOLDER = _join('segmantation_program')
TEST_FOLDER = _join('tests')
'''
