from random import randint

intArray = []
binArray = []

#converts a decimal number to a binary number
def decimalToBinary(dec, numberOfBits):
    answer = []
    while(dec > 0):
        answer.append(dec % 2)
        dec = dec // 2
    zeros = []
    for i in range(0, numberOfBits-len(answer)):
        zeros.append(0)
    return zeros + answer

#converts a binary number to a decimal number
def binaryToDecimal(bin):
    place = 1
    sum = 0
    for i in range(len(bin)-1, -1, -1):
        sum += (place * bin[i])
        place *= 2
    return sum

#check if an integer is prime
def isPrime(n):
  if n == 1 or n == 2 or n == 3:
      return True
  if n < 2 or n%2 == 0:
      return False
  if n < 9:
      return True
  if n%3 == 0:
      return False
  r = int(n**0.5)
  f = 5
  while f <= r:
    if n%f == 0: return False
    if n%(f+2) == 0: return False
    f +=6
  return True

#fills an array with number from 1 to 10
def createArray(n):
    for i in range(1,n):
        intArray.append(i)


if __name__ == '__main__':
    # turns an array of decimals numbers into its equivalent binary representation
    createArray(10)
    for i in range(1, len(intArray)):
        x = decimalToBinary(intArray[i], 8)
        binArray.append(x)

    for i in range(1, 10):
        if isPrime(i) == True:
            print(i, " is prime")
        else:
            print(i, " is NOT prime")