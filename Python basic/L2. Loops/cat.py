# Ctrl + / : multi-line comment
# Shift + Alt + A : block comment

# while loop - while condition is a expression to maintain the loop
i = 0
while i < 3:
    print("meow")
    i += 1

# for loop - more concise and clear than while loop
for _ in range(3):
    print("meow")

for i in [0, 1, 2]:
    print("meow")

# pythonic style
print("meow\n" * 3, end="")  # end override \n

# prompt the user to input
while True:
    n = int(input("What's n? "))
    if n > 0:
        break

for _ in range(
    n
):  # if the iterative element (etc. i, j, k) will not be used, use _ to be pythonic
    print("meow")


# capuslate them into function
def main():
    number = get_number()
    meow(3)


def get_number():
    while True:
        n = int((input("What's n? ")))
        if n > 0:
            return n


def meow(n):
    for _ in range(n):
        print("meow")


if __name__ == "__main__":
    main()
