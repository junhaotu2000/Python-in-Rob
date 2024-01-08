x = int(input("What's x ? "))
y = int(input("What's y ? "))


# if - if - if for most intuitive way (you need mutli-exclusive condition)
if x < y:  # use boolean input
    print("x is less than y")  # indentation
if x > y:
    print("x is greater than y")
if x == y:
    print("x is eqaul to y")

# if - elif - elif for many judgements at same time
if x < y:  # use boolean input
    print("x is less than y")  # indentation
elif x > y:
    print("x is greater than y")
elif x == y:
    print("x is eqaul to y")

# if - elif - else for efficient approach
if x < y:  # use boolean input
    print("x is less than y")  # indentation
elif x > y:
    print("x is greater than y")
else:
    print("x is eqaul to y")

# or to combine first two judgement
if x < y or x > y:
    print("x is not equal to y")
else:
    print("x is equal to y")

# change conditional expression to make it clear and concise
if x != y:
    print("x is not equal to y")
else:
    print("x is equal to y")
