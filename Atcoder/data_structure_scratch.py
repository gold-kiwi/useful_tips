class Stack:
    def __init__(self,data = []):
        self.data = data
    def push(self,x):
        self.data.append(x)
        return self.data
    def pop(self):
        if len(self.data) <= 0:
            print('Queue is Empty')
            return -1
        else:
            out = self.data.pop()
            return out
    def show(self):
        print(self.data)

    def list(self):
        return self.data

class Queue:
    def __init__(self,data = []):
        self.data = data
    def enqueue(self,x):
        self.data.append(x)
        return self.data
    def dequeue(self):
        if len(self.data) <= 0:
            print('Queue is Empty')
            return -1
        else:
            out = self.data[0]
            del self.data[0]
            return out
    def show(self):
        print(self.data)

    def list(self):
        return self.data

class Heap():
    def __init__(self,data = []):
        self.data = data
        if len(self.data) > 1:
            self.build_min_heap()

    def min_heapify(self,i,data):
        if len(data) > 1:
            left_index = i * 2 + 1
            right_index = i * 2 + 2
            if data[i] > data[left_index]:
                data[i],data[left_index] = data[left_index],data[i]
            
            if right_index < len(data): #right child exists.
                if data[i] > data[right_index]:
                    data[i],data[right_index] = data[right_index],data[i]
            
    def build_min_heap(self):
        for i in reversed(range(len(self.data)//2)):
            self.min_heapify(i,self.data)

    def push(self,x):
        self.data.append(x)
        self.build_min_heap()

    def pop(self):
        self.data[0], self.data[-1] = self.data[-1], self.data[0]
        min_val = self.data.pop()
        self.build_min_heap()
        return min_val

    def show(self):
        print(self.data)

    def get_data(self):
        return self.data

    def sort(self):
        data_for_sort = self.data.copy()

        sorted_list = []
        for _ in range(len(data_for_sort)):
            data_for_sort[0], data_for_sort[-1] = data_for_sort[-1], data_for_sort[0]
            sorted_list.append(data_for_sort.pop())
            self.min_heapify(0,data = data_for_sort)

        self.data = sorted_list.copy()
        return sorted_list


def quick_sort(data):
    if len(data) <= 1:
        return data
    elif len(data) == 2:
        if data[0] > data[1]:
            data[0],data[1] = data[1],data[0]
        return data
    else:
        pivot = data[0]
        left_data = []
        right_data = []
        for i in range(1,len(data)):
            if data[i] < pivot:
                left_data.append(data[i])
            else:
                right_data.append(data[i])
            
        sorted_left_data = quick_sort(left_data)
        sorted_right_left = quick_sort(right_data)
        return sorted_left_data + [pivot] + sorted_right_left


def merge_sort(data):
    if len(data) <= 1:
        return data
    elif len(data) == 2:
        if data[0] > data[1]:
            data[0],data[1] = data[1],data[0]
        return data
    else:
        mid = len(data)//2
        left_data = data[:mid]
        right_data = data[mid:]
        sorted_left_data = merge_sort(left_data)
        sorted_right_left = merge_sort(right_data)

        sorted_data = []
        while True:
            

        return sorted_left_data + sorted_right_left



def main():
    print('study data structure')

    data = [5,4,6,1,2,7,3]
    print(data)
    print(quick_sort(data))
    print(merge_sort(data))

if __name__ == '__main__':
    main()