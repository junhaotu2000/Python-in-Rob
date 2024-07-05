def hello(to="world"):
    print("hello,", to)


def main():
    hello()  # use default input
    name = input("What's your name?")
    hello(name)  # use assigned input


def main2():
    x = int(input("What's x ?"))
    print("x squared is", square(x))


def square(n):
    return pow(n, 2)  # n**2: py style for power opt


if __name__ == "__main__":
    main()
