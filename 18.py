# return max profit for stock
# https://www.cnblogs.com/SupremeBoy/articles/14255365.html
# https://blog.csdn.net/qq_44861675/article/details/115261558
# https://www.cnblogs.com/SupremeBoy/articles/14255365.html
# https://www.cnblogs.com/simplekinght/p/13190016.html

def maxProfit(array):
    if len(array) < 2:
        return 0
    
    # DP with space optimization
    # `cash` is the max profit if we don't hold a stock on the current day
    # `hold` is the max profit if we do hold a stock on the current day
    
    # Base cases for day 0
    cash = 0
    hold = -array[0] # We "buy" the stock on day 0
    cashArray = []
    holdArray = []
    cashArray.append(cash)
    holdArray.append(hold)
    
    for i in range(1, len(array)):
        prev_cash = cash
        # Max profit with cash today: either hold cash from yesterday, or sell stock from yesterday
        cash = max(cash, hold + array[i])
        # Max profit with stock today: either hold stock from yesterday, or buy stock with cash from yesterday
        hold = max(hold, prev_cash - array[i])

        cashArray.append(cash)
        holdArray.append(hold)
        
    # The final answer must be `cash`, as it's always better to sell than to hold on the last day.
    return cash

# array = [7, 1, 5, 3, 6, 4]
array = [7, 2, 6, 1, 7, 2]
result = maxProfit(array) #should be 7 as 5-1 + 6-3 = 7
print(result)

array = [7, 1, 5, 18, 6, 4]
result = maxProfit(array) #should be 17 as 18 - 1 = 17
print(result)