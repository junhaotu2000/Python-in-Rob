import random


class Hat:
    def __init__(self):
        self.houses = ["Gryffindor", "Hullepuff", "Ravenclaw", "Slytherin"]

    def sort(self, name):
        print(name, "is in", random.choice(self.houses))
