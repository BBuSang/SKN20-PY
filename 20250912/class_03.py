class Product():
    serial_num = 0
    def __init__(self):
        Product.serial_num += 1
        self.serial_num = Product.serial_num
        self.name = None


tv1 = Product()
tv2 = Product()
tv3 = Product()

print(tv1.serial_num)
print(tv2.serial_num)
print(tv3.serial_num)