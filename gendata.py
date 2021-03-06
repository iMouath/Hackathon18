import csv
import random

data = []
data_binary = []


def write():
    with open('dataset.csv', 'w+') as csvfile:
        filewriter = csv.writer(csvfile, )
        filewriter.writerow(data)


def gendata():
    for i in range(0, 100):
        num = random.randint(0, 50)
        binary_num = bin(num)
        data.append(num)
        data_binary.append(binary_num)


def is_prime(n):
    if n == 2 or n == 3: return True
    if n < 2 or n % 2 == 0: return False
    if n < 9: return True
    if n % 3 == 0: return False
    r = int(n ** 0.5)
    f = 5
    while f <= r:
        print('\t', f)
        if n % f == 0:
            return False
        if n % (f + 2) == 0:
            return False
        f += 6
    return True


if __name__ == '__main__':
    gendata()
    for i in data:
        print(is_prime(i))