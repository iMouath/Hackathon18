from random import randint
import csv

intArray = []
binArray = []
data = []
data_binary = []


# converts a decimal number to a binary number
def decimalToBinary(dec, numberOfBits):
    answer = ""
    while (dec > 0):
        answer = str(dec % 2) + answer
        dec = dec // 2
    zeros = ""
    for i in range(0, numberOfBits - len(answer)):
        zeros = zeros + '0'
    binaryString = zeros + answer
    return binaryString


# converts a binary number to a decimal number
def binaryToDecimal(bin):
    place = 1
    sum = 0
    for i in range(len(bin) - 1, -1, -1):
        sum += (place * bin[i])
        place *= 2
    return sum


# check if an integer is prime
def isPrime(n):
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n ** 0.5)
    f = 5
    while f <= r:
        if n % f == 0: return False
        if n % (f + 2) == 0: return False
        f += 6
    return True


# creates a useable data set with specified number
def createUseableDate(int, numOfBits):
    primeNumber = 0
    if (isPrime(int) == True):
        primeNumber = 1
    string = decimalToBinary(int, numOfBits + 1)
    useableData = []
    for i in range(1, len(string)):
        if (string[i] == '1'):
            useableData.append(0.9)
        else:
            useableData.append(0.1)
    # return useableData
    if (primeNumber == 1):
        useableData.append(1)
    else:
        useableData.append(0)
    return useableData


def write(dataToWrite):
    with open('prime_test.csv', 'w+') as csvfile:
        for i in dataToWrite:
            str_line = str(useableData[i])
            str_line = str_line.replace("[", "", 1)
            str_line = str_line.replace("]", "", 1)
            csvfile.write(dataToWrite)
            csvfile.write('\n')


if __name__ == '__main__':

    numArray = []
    useableData = []
    for x in range(1, 5001):
        numArray.append(randint(1, 100000))
    # print(numArray)
    for i in range(0, len(numArray)):
        num = numArray[i]
        useableData = createUseableDate(num, 24)
        string = str(useableData)
        string = string.replace("[", "", 1)
        string = string.replace("]", "", 1)
        print(string)