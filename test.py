from typing import List


class Test:
    def __init__(self, a, b):
        self.b = b
        self.a = a


    def __getattr__(self, item):
        return item

def foo() -> List[bool]:
    return [1,2,3]

print(Test(1, 2).c)
print(foo())