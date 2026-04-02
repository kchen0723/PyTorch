def GetPermutationFromUniqueArray(arr: list, k: int):
    result = []
    if len(arr) < k:
        return result
    candidate = []
    select = [False] * len(arr)
    GetPermutationFromUniqueArrayHelp(arr, k, select, candidate, result)
    return result

def GetPermutationFromUniqueArrayHelp(arr: list, k: int, select: list, candidate: list, result: list):
    if len(candidate) == k:
        result.append(candidate.copy())
        return 
    for i in range(len(arr)):
        if select[i]:
            continue
        select[i] = True
        candidate.append(arr[i])
        GetPermutationFromUniqueArrayHelp(arr, k, select, candidate, result)
        candidate.pop()
        select[i] = False

arr = [1, 2, 3]
result = GetPermutationFromUniqueArray(arr, 2)
print(result)