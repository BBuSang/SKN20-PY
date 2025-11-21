# 클래스
# 클래스 변수, 인스턴스 변수
# 생성자 __init__
# 메소드 __str__ __eq__ __lt__ __le__ __gt__ __ge__ __ne__
# property getter, setter, deleter, private --> 함수를 변수처럼 사용
# 객체생성

# 상품관리 클래스명 Product
# 상품명 Product_name, 가격 Product_price, 재고 Product_stock

class Product() : 
    count = 0
    def __init__(self, product_name, product_price, product_stock):
        self.id = Product.count+1
        Product.count +=1
        self.product_name = product_name
        self._product_price = product_price
        self._product_stock = product_stock


    @property
    def product_price(self):
        return self._product_price
    @property
    def product_stock(self):
        return self._product_stock
    @product_stock.setter
    def product_stock(self, value):
        if value < 0 :
            raise ValueError("재고는 음수가 될 수 없습니다.")
        self._product_stock = value
    @product_price.setter
    def product_price(self, value):
        if value < 0 :
            raise ValueError("가격은 음수가 될 수 없습니다.")
        self._product_price = value
    
    def price_down(self, name, persent):
        if self.product_name == name :
            self.product_price = int(self.product_price * (1 - persent/100))

    def price_up(self, name, persent):
        if self.product_name == name :
            self.product_price = int(self.product_price * (1 + persent/100))

    def total_stock(self):
        print(f'현재 모든 제품의 수량의 합 : {sum([a.product_stock for a in products])}')

    def remove(self):
        if self.product_stock > 0 :
            print("재고가 남아 있어 삭제할 수 없습니다.")
        else :
            print(f'{self.product_name} 제품이 삭제되었습니다.')
            del self

    def __str__(self):
        return f'상품ID : {self.id}, 상품명 : {self.product_name}, 가격 : {self.product_price}, 재고 : {self.product_stock}'
    def __eq__(self, value):
        return self.product_stock == value.product_stock
    def __ne__(self, value):
        return self.product_stock != value.product_stock
    def __lt__(self, value):
        return self.product_stock < value.product_stock
    def __le__(self, value):
        return self.product_stock <= value.product_stock
    def __gt__(self, value):
        return self.product_stock > value.product_stock
    def __ge__(self, value):
        return self.product_stock >= value.product_stock

    def stock_up(self, value):
        self.product_stock += value

products = [
    Product("노트북", 2000000, 15),
    Product("스마트폰", 1000000, 10),
    Product("태블릿", 900000, 5)
]
# 노트북의 가격을 20% 인하
# 스마트폰은 가격을 10% 인상
for p in products :
    p.price_down("노트북", 20)
    p.price_up("스마트폰", 10)
# 제품 출력
    print(p)
# 제품 추가
products.append(Product("모니터", 400000, 20))
# 제품 삭제 - 수량이 남아 있으면 삭제 못하게
products[2].remove()
# 현재 모든 제품의 수량의 합
products[0].total_stock()
# 수량을 기준으로 같다 크다 크거나 같다 작다 작거나 같다
print(products[0] == products[1])
print(products[0] != products[1])
print(products[0] < products[1])
print(products[0] <= products[1])
print(products[0] > products[1])
print(products[0] >= products[1])

# 수량을 증가
products[2].stock_up(-5)
products[2].remove()
print(products[2])