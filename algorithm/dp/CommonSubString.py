import numpy as np

def GetLongestCommonSubStringByDp(left: str, right: str):
    if not left or not right:
        return ""
    dp = np.zeros((len(left) + 1, len(right) + 1), dtype=int)
    max_length, end_index = 0, 0
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                dp[i][j] = dp[i -1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i

    result = left[end_index - max_length: end_index]
    return result

result = GetLongestCommonSubStringByDp("maven", "having")
print(result)