
def main():
    x = int(input("What's x ? "))
    if x % 2 == 0:
        print("Even")
    else:
        print("Odd")

def is_even(n):
    if  n % 2 == 0:
        return True
    else:
        return False
    
# condense many lines into single line
def is_even1(n):
    return True if n % 2 == 0 else False

# the most concise way -- directly output condition judgement
def is_even2(n):
    return (n % 2==0)

if __name__ == "__main__":
    main()
