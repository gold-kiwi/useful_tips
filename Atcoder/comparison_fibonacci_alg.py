# coding: utf-8
#メモ化再帰の威力を確かめるために、メモ化再帰ありなしのフィボナッチ数計算の処理時間の比較を行う
import sys
import time

sys.setrecursionlimit(2048)

def fibonacci(n):
    if n == 1:
        return 1
    elif n > 1:
        return fibonacci(n-1) + fibonacci(n-2)
    else:
        return 0

class fibonacci_DP():
    def __init__(self, ):
        self.fib_memo = {}
        self.fib_memo[0] = 0
        self.fib_memo[1] = 1

    def fibonacci(self,n):
        if n > 1:
            if not n in self.fib_memo:
                self.fib_memo[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self.fib_memo[n]

    def show_fib_memo(self):
        print(self.fib_memo)
    
    def get_fib_memo(self):
        return self.fib_memo

    def calc_and_show_fib_memo(self,n=1):
        print('n = %d',n)
        fibonacci(n)
        print(self.fib_memo)

    def clear_fib_memo(self):
        self.fib_memo = {}


def main():
    args = sys.argv
    if len(args) >= 2:
        fib_len = int(args[1])
    else:
        fib_len = 20

    print('calc fibonacci %d' % fib_len)

    start_time = time.time()
    fib_list = {}
    for i in range(fib_len + 1):
        fib_n = fibonacci(i)
        fib_list[i] = fib_n
    print(fib_list)
    end_time = time.time()
    print ("processing_time:{:f}".format(end_time-start_time) + "[sec]")

    start_time = time.time()
    fib_DP = fibonacci_DP()
    for i in range(fib_len + 1):
        fib_DP.fibonacci(i)
    fib_DP.show_fib_memo()
    end_time = time.time()
    print ("processing_time:{:f}".format(end_time-start_time) + "[sec]")

if __name__ == '__main__':
    main()