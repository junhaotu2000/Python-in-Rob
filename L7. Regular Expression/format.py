import re

name = input("What's your name?").strip()
matches = re.search(r"^(.+), (.+)$", name)

# a comman way of using group func
if matches:
    last, first = matches.groups()
    name = f"{first} {last}"
    # name = matches.group(2) + " " + matches.group(1)
    print(f"hello, {name}")

# := - is right side is true then, then assign value to left side
if matches := re.search(r"^(.+), (.+)$", name):
    name = matches.group(2) + " " + matches.group(1)
    print(f"hello, {name}")
