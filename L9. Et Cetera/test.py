def meow(n: int) -> None:  # None is a return hints
    """_summary_

    Args:
        n (int): _description_
    """
    for _ in range(n):
        print("meow")


number: int = int(input("Number: "))
meows: str = meow(number)

print(meows)
