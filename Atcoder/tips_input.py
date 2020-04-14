# -*- coding: utf-8 -*-

#文字列入力
s = input()

#整数入力
n = int(input())

# スペース区切りの整数の入力
a, b = map(int, input().split())
a, b, c = map(int, input().split())

#整数1次元配列（リスト）
t = list(map(int, input().strip().split()))

#整数2次元配列
p = []
for i in range(n):
    t = list(map(int, input().strip().split()))
    p.append(t)
