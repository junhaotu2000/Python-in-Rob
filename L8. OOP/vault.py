class Vault:
    def __init__(self, galleons = 0, sickles = 0, kunts = 0) -> None:
        self.galleons = galleons
        self.sickles = sickles
        self.kunts = kunts
    
    def __str__(self):
        return f"{self.galleons}, {self.sickles}, {self.kunts}"
    
    def __add__(self, other):
        galleons = self.gallens + other.galleons
        sickles = self.sickles + other.sickles
        kunts = self.kunts + other.kunts
        return Vault(galleons, sickles, kunts)
                
potter = Vault(100, 50, 25)
print(potter)

weasley = Vault(25, 50, 100)
print(weasley)

# direct add - TypeError: unsupported operand type(s) for +: 'Vault' and 'Vault'
# overloading add to resolve this issue
total = potter + weasley





