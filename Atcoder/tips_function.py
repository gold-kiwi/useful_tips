# -*- coding: utf-8 -*-


def binary(n):
    return bin(n)[2:]

def pow_w_mod(a, x, mod):  # a^x mod n
    x = [int(b) for b in binary(x)]
    y = a
    for i in range(1, len(x)):
        y = (y**2) % mod
        if x[i] == 1:
            y = (y * a) % mod
    return y

 # mを法とするaの乗法的逆元
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def gcd(a,b):
    while b!=0:
        a,b=b,a%b
    return a

def lcm(a,b):
    return a*b//gcd(a,b)

def egcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y

# nCr mod m
# modinvが必要
# rがn/2に近いと非常に重くなる
def combination(n, r, mod=10**9 + 7):
    r = min(r, n - r)
    res = 1
    for i in range(r):
        res = res * (n - i) * modinv(i + 1, mod) % mod
    return res

def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)

    # divisors.sort()
    return divisors

def factorization(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])

    if temp!=1:
        arr.append([temp, 1])

    if arr==[]:
        arr.append([n, 1])

    return arr