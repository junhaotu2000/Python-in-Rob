import re

email = input("What's your email? ").strip()
""" 
--   . any character except a newline
--   * 0 or more repetitions
--   + 1 or more repetitions
--   ? 0 or 1 repetition
--   {m} m repetitions
--   {m, n} m-n repetitions
"""
if re.search(r"^\w+@\w+\.edu$", email):
    print("Valid")
else:
    print("Invalid")
