# list
students = ["Herione", "Harry", "Ron", "Draco"]
houses = ["Gryffindor", "Gryffindor", "Gryffindor", "Slytherin"]

# call variable if it will be used in loop
for name in students:
    print(name)

# list do not have shape, but len to get length of list
for i in range(len(students)):
    print(i + 1, students[i], ":", houses[i])

# dict
students = {
    "Herione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}

for student in students:
    print(student, students[student], sep=", ")  # student only return all of the key

# list of dict
students = [
    {"name": "Hermione", "house": "Gryffindor", "patronus": "Otter"},
    {"name": "Harry", "house": "Gryffindor", "patronus": "Stag"},
    {"name": "Ron", "house": "Gryffindor", "patronus": "Jack Russell terrier"},
    {"name": "Draco", "house": "Slytherin", "patronus": None},
]

for i, student in enumerate(students):
    print(i + 1, student["name"], student["house"], student["patronus"], sep=", ")
