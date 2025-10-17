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

def maxProfitBuyOnlyOnceDp(arr: list):
    if len(arr) < 2:
        return 0
    
    # base case
    holdCash = 0
    holdStock = 0 - arr[0]
    dp = [[holdCash, holdStock]]

    for i in range(1, len(arr)):
        holdCash = max(dp[i - 1][0], dp[i - 1][1] + arr[i])  #sell the stock
        holdStock = max(dp[i - 1][1], 0 - arr[i]) #buy the stock
        dp.append([holdCash, holdStock])

    return dp[len(arr) - 1][0]

arr = [7, 1, 5, 3, 6, 4]
result = maxProfitBuyOnlyOnce(arr)
print(result)
result = maxProfitBuyOnlyOnceDp(arr)
print(result)

# 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
# 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
# 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票
def maxProfitBuyMultipleTimes(arr: list):
    if len(arr) < 2:
        return 0
    
    # base case
    holdCash = 0
    holdStock = 0 - arr[0]
    dp = [[holdCash, holdStock]]

    for i in range(1, len(arr)):
        holdCash = max(dp[i - 1][0], dp[i - 1][1] + arr[i])  #sell the stock
        holdStock = max(dp[i - 1][1], dp[i - 1][0] - arr[i]) #buy the stock
        dp.append([holdCash, holdStock])

    return dp[len(arr) - 1][0]
    
arr = [3, 2, 6, 5, 0, 3]
result = maxProfitBuyMultipleTimes(arr)
print(result)