# create a function that calls all tests 
# use pytest 
# pytest.main()

import pytest

def test_all():
    # pytest, folder is tests 
    pytest.main(['tests'])