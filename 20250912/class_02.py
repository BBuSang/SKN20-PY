class People():
    def make_instance(self):
        self.name = None
        self.age = None
        self.addr = None

h1 = People()
h2 = People()
# h1.make_instance()
h1.addr = 1

print(h1.addr)