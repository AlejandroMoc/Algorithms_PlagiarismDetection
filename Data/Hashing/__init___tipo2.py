def lnxzr(input_lst):
    unique_lst = []
    for elem in input_lst:
        if elem not in unique_lst:
            unique_lst.append(elem)
    return unique_lst

class AxclQr:
    def __init__(self, name):
        self.name = name

    def shw_nm(self):
        print(f"My name is {self.name}")

def cncat_str(str1, str2):
    return str1 + str2

lst1 = [1, 2, 2, 3, 4, 4, 5]
print(lnxzr(lst1))

obj1 = AxclQr("Alice")
obj1.shw_nm()

str1 = "Hello"
str2 = "World"
print(cncat_str(str1, str2))