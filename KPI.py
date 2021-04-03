
def KPI():
    data = open('Record.csv', encoding='utf-8-sig').readlines()
    data = [ i.strip('\n').split(',') for i in data ]

    # 紀錄每筆損益
    Profit = []
    for i in data:
        # 多單
        if i[1] == 'B':
            P = (float(i[7])-float(i[4]))*float(i[8])
            Profit.append(P)
        # 空單
        elif i[1] == 'S':
            P = (float(i[4])-float(i[7]))*float(i[8])
            Profit.append(P)

    # 獲利及虧損的資料
    Win = [ i for i in Profit if i > 0 ]
    Loss = [ i for i in Profit if i < 0 ]

    # 總損益
    Total_Profit = sum(Profit)
    # 總交易次數
    Total_Trade = len(Profit)
    # 平均損益
    Avg_Profit = Total_Profit / Total_Trade
    # 勝率
    Win_Rate = len(Win) / Total_Trade
    # 獲利因子及賺賠比
    if len(Loss) == 0:
        Profit_Factor = 'NA'
        Win_Loss_Rate = 'NA'
    else:
        Profit_Factor = sum(Win) / -sum(Loss)
        Win_Loss_Rate = (sum(Win)/len(Win)) / (-sum(Loss)/len(Loss))
    # 最大資金回落
    MDD,Capital,MaxCapital = 0,1000000,1000000
    for p in Profit:
        Capital += p
        MaxCapital = max(MaxCapital,Capital)
        DD = MaxCapital - Capital
        MDD = max(MDD,DD)
    # 每筆報酬率
    Return_Rate = []
    Capital = 1000000
    for p in Profit:
        ret = p / Capital
        Return_Rate.append(ret)
        Capital += p
    # 夏普比率 (無法取得每日報酬率，因此忽略無風險利率)
    import numpy as np
    Return_Rate = np.array(Return_Rate)
    Sharpe_Ratio = np.mean(Return_Rate) / np.std(Return_Rate)

    # 將績效指標匯出
    file = open('KPI.csv','w', encoding='utf-8-sig')
    file.write(','.join(['總損益','總交易次數','平均損益','勝率','獲利因子','賺賠比','最大資金回落','夏普比率','\n']))
    file.write(','.join([str(Total_Profit),str(Total_Trade),str(Avg_Profit),str(Win_Rate),str(Profit_Factor),str(Win_Loss_Rate),str(MDD),str(Sharpe_Ratio)]))
    file.close()
