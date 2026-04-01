def GetMinCover(source: str, target: str):
    if len(source) < len(target):
        return ""
    targetDictionary = buildDictonary(target)
    left, right, start, matchCount = 0, 0, 0, 0
    length = len(source)
    window = {}
    while right < len(source):
        item = source[right]
        right += 1
        if item in targetDictionary:
            if item in window:
                window[item] += 1
            else:
                window[item] = 1
            if window[item] == targetDictionary[item]:
                matchCount += 1
        
        while matchCount == len(targetDictionary):
            if right - left + 1 < length:
                length = right - left + 1
                start = left
            removeCandidate = source[left]
            left += 1
            if removeCandidate in window:
                if window[removeCandidate] == targetDictionary[removeCandidate]:
                    matchCount -= 1
                window[removeCandidate] -= 1
    if length < len(source):
        return source[start: start + length]
    return ""

def buildDictonary(target: str):
    dic = {}
    for c in target:
        if c in dic:
            dic[c] += 1
        else:
            dic[c] = 1
    return dic

if __name__ == "__main__":
    source = "GetMinCover"
    target = "Mov"
    result = GetMinCover(source, target)
    print(result)