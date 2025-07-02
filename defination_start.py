class Book:
    def __init__(self, title, author, pages, price):
        self.title = title
        self.author = author
        self.pages = pages
        self.price = price
        self.__secret = "This is a secret attribute"

    def set_discount(self, amount):
        self._discount = amount

    def get_price(self):
        if hasattr(self, "_discount"):
            return self.price - (self.price * self._discount)
        else:
            return self.price

# Creating book objects
b1 = Book("War and Peace", "Leo Tolstoy", 1225, 39.95)
b2 = Book("The Catcher in the Rye", "JD Salinger", 234, 29.95)

# Getting original prices
print(b1.get_price())
print(b2.get_price())

# Applying discount to b2
b2.set_discount(0.25)
print(b2.get_price())

# Accessing the protected attribute (not recommended, but possible)
print(b2._discount)

# Accessing the private attribute (name mangled)
print(b2._Book__secret)
