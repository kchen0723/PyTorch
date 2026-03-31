def IterationArray(arr: list):
    if len(arr) == 0:
        return
    for i in range(len(arr)):
        print(arr[i])

def RecursionArray(arr: list):
    if len(arr) == 0:
        return
    RecursionArrayHelp(arr, 0)

def RecursionArrayHelp(arr:list, index: int):
    if index >= len(arr):
        return
    print(arr[index])
    RecursionArrayHelp(arr, index + 1)

arr = [3, 2, 6, 5, 0, 3]
IterationArray(arr)
RecursionArray(arr)