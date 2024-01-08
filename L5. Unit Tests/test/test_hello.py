from hello import hello


def test_default():
    assert hello() == "hello, world"


def test_argument():
    assert hello("David") == "hello, David"


# use for loop to test multiply cases
def test_arguments():
    for name in ["Hermione", "Harry", "Ron"]:
        assert hello(name) == f"hello, {name}"
