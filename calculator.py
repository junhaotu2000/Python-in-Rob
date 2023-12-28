
# convert the type at the input stage 
x = int(input("What's x?"))
y = int(input("What's y?"))

x = float(input("What's x?"))
y = float(input("What's y?"))

# round(number[, ndigits]) - []: sometimes this means optional
# round the result into int 
z = round(x + y)

# round the result into two decimals
z = round(x/y, 2)

# add comman , into every thousand num
print(f"{z:,}")

# use f-string to retain the values into two decimals
print(f"{z:.2f}")