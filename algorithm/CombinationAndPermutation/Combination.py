def GetCombintionFromUniqueArray(arr: list, k: int):
    result = []
    if len(arr) < k:
        return result
    candidate = []
    GetCombintionFromUniqueArrayHelp(arr, k, 0, candidate, result)
    return result

def GetCombintionFromUniqueArrayHelp(arr: list, k: int, index: int, candidate: list, result: list):
    if len(candidate) == k:
        result.append(candidate.copy())
        return
    for i in range(index, len(arr)):
        candidate.append(arr[i])
        GetCombintionFromUniqueArrayHelp(arr, k, i + 1, candidate, result)
        candidate.pop()

def GetCombintionFromDuplicatedArray(arr: list, k: int):
    result = []
    if len(arr) < k:
        return result
    candidate = []
    arr.sort()
    GetCombintionFromDuplicatedArrayHelp(arr, k, 0, candidate, result)
    return result

def GetCombintionFromDuplicatedArrayHelp(arr: list, k: int, index: int, candidate:list, result: list):
    if len(candidate) == k:
        result.append(candidate.copy())
        return
    for i in range(index, len(arr)):
        if (i > index and arr[i] == arr[i - 1]):
            continue
        candidate.append(arr[i])
        GetCombintionFromDuplicatedArrayHelp(arr, k, i + 1, candidate, result)
        candidate.pop()

def GetCombintionMultipleTime(arr: list, k:int):
    result = []
    if len(arr) < k:
        return result
    candidate = []
    GetCombintionMultipleTimeHelp(arr, k, 0, candidate, result)
    return result

def GetCombintionMultipleTimeHelp(arr: list, k: int, index: int, candidate: list, result: list):
    if len(candidate) == k:
        result.append(candidate.copy())
        return
    for i in range(index, len(arr)):
        candidate.append(arr[i])
        GetCombintionMultipleTimeHelp(arr, k, i, candidate, result)
        candidate.pop()

arr = [1, 2, 3]
result = GetCombintionFromUniqueArray(arr, 2)
print(result)

result = GetCombintionMultipleTime(arr, 2)
print(result)

arr = [1, 2, 2, 3]
result = GetCombintionFromDuplicatedArray(arr, 2)
print(result)