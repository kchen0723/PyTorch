# 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
# 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
# 注意：你不能在买入股票前卖出股票。
# 示例 1:
# 输入: [7,1,5,3,6,4]
# 输出: 5

def maxProfitBuyOnlyOnce(arr: list):
    if len(arr) < 2:
        return 0
    
    dp = [0]
    minPrice = arr[0]
    for i in range(1, len(arr)):
        minPrice = min(minPrice, arr[i])
        profit = max(dp[i - 1], arr[i] - minPrice)
        dp.append(profit)

    return dp[len(arr) - 1]
    
arr = [7, 1, 5, 3, 6, 4]
result = maxProfitBuyOnlyOnce(arr)
print(result)