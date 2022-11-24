import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import graphviz

from scipy.stats import rankdata
import scipy.stats as stats

from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
# import baostock as bs
# from ta.volume import VolumeWeightedAveragePrice
import statsmodels.api as sm
from scipy.stats.mstats import zscore
# import talib

from scipy.stats import spearmanr
import datetime
from numba import jit
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import math
import cloudpickle
import matplotlib.pyplot as plt

# @jit
def _factor_backtest(factor_perd, market_price):
    pred = pd.Series(factor_perd.flatten()).fillna(0)
    evaluation = []
    slippage = 0
    shares = 1
    comission = 0.0003
     
    backtest_data = market_price
    trades = pred

    short_open = 0
    long_open = 0
    held_long = 0
    held_short = 0
    profit = []
    profit_temp = 0
    initial_assets = shares*max(backtest_data.open.values)
    initial_cash = shares*max(backtest_data.open.values)
    initial_assets = 10000
    initial_cash = 10000
    net_worth = [initial_cash]
    
    for i in range(len(trades)):
        current_pred = trades.iloc[i]
        current_close = backtest_data.iloc[i].open.astype('float')
        #open long
        if current_pred>=0.75 and held_long==0 and held_short==0:
            held_long = 1
            held_short = 0
            long_open = current_close+slippage
            short_open = 0
            #print('open long')
                    
        #hold long
        elif current_pred>=0.75 and held_long==1 and held_short==0:
            #print('hold long')
            pass
                
            #open long and close short
        elif current_pred>=0.75 and held_long==0 and held_short==1:
            held_long = 1
            #close short and calculate profit
            held_short = 0
            profit_temp = (short_open-(current_close+slippage))*shares*(1-comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            #open long
            short_open = 0
            long_open = current_close+slippage
            #print('open long and close short')
                    
        #open short
        elif current_pred<=-0.75 and held_long==0 and held_short==0:
            held_long = 0
            held_short = 1
            long_open = long_open
            short_open = current_close+slippage
            profit = profit
            #print('open short')
            
        #keep short
        elif current_pred<=-0.75 and held_long==0 and held_short==1:
            #print('keep short')
            pass
                
        #close long and open short
        elif current_pred <=-0.75 and held_long==1 and held_short==0:
            #close long
            held_long = 0
            held_short = 1
            profit_temp = ((current_close-slippage)-long_open)*shares*(1-comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            #open short
            long_open = 0
            short_open = current_close-slippage
            profit = profit
            #print('close long and open short')
                    
        #closeout long
        elif current_pred<0.75 and current_pred>-0.75 and held_long==1 and held_short==0:
            held_long = 0
            held_short = 0
            profit_temp = ((current_close-slippage)-long_open)*shares*(1-comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            short_open = 0
            long_open = 0
            #print('closeout long')        
            
        #closeout short    
        elif current_pred<0.75 and current_pred>-0.75 and held_long==0 and held_short==1:
            held_long = 0
            held_short = 0
            profit_temp = (short_open-(current_close+slippage))*shares*(1-comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            short_open = 0
            long_open = 0
            #print('closeout short')
            
    total_return = (initial_cash-initial_assets)/initial_assets
    print('总收益率', total_return)
    shaprpe_df = pd.Series(profit)
    sharpe_temp = (shaprpe_df - shaprpe_df.shift(1))/shaprpe_df.shift(1)
    sharpe = sharpe_temp.mean()/sharpe_temp.std()*np.sqrt(len(profit))
    
    a = np.maximum.accumulate(net_worth)
    l = np.argmax((np.maximum.accumulate(net_worth) - net_worth)/np.maximum.accumulate(net_worth))
    k = np.argmax(net_worth[:l])
    max_draw = (net_worth[k] - net_worth[l])/(net_worth[l]) 
    print('最大回撤', max_draw)        
        
    win_count = 0
    loss_count = 0
    initial_profit = 0
    for i in range(len(net_worth)):
        current_profit = net_worth[i]
        if i==0:
            if current_profit > initial_assets:
                win_count += 1
            else:
                loss_count += 1
        else:
            last_profit = net_worth[i-1]
            if current_profit > last_profit:
                win_count += 1
            else:
                loss_count += 1
    win_rate = win_count/len(net_worth)
    print('胜率', win_rate)
    
    total_gain = 0
    total_loss = 0
    for i in range(len(profit)):
        if profit[i] >0:
            total_gain += profit[i]
        else:
            total_loss += profit[i]
    gain_loss_ratio = (total_gain/win_count)/(abs(total_loss)/loss_count)
    print('盈亏比', gain_loss_ratio)
    
    result = total_return*np.nan_to_num(sharpe,nan=1)*win_rate*gain_loss_ratio*(1-max_draw)
    
    x = np.array(net_worth).reshape(len(net_worth),)
    y = np.arange(len(net_worth))
    
    plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    fig = plt.figure(figsize=(16,9))
    # plt.plot(y, x)
    # plt.title('因子资金曲线',fontsize=20)
    # plt.xlabel('交易次数',fontsize=20)
    # plt.ylabel('账户净值',fontsize=20)
    # plt.savefig('sample.png')
    # plt.cla()
    return pd.Series(net_worth)
    # return result

def alpha_factor_graph(num, best_programs_dict):
    # 打印指定num的表达式图

    factor = best_programs[num-1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)

    return graph


target_inv = 'PTA'
start_train_date = '2022-03-01'
end_train_date = '2022-06-01'
start_test_date = end_train_date
end_test_date = '2022-07-01'
n_thread = 32
target_inv_list = [
        ('2022-03-01', '2022-06-01', '2022-07-01', 'PTA'), \
        ('2021-04-01', '2022-04-01', '2022-07-01', 'PTA'), \
        ('2021-01-01', '2022-01-01', '2022-07-01', 'PTA'), \
        ('2020-01-01', '2022-01-01', '2022-07-01', 'PTA'), \
        # ('2022-03-01', '2022-06-01', '2022-07-01', 'RB'), \
        # ('2021-04-01', '2022-04-01', '2022-07-01', 'RB'), \
        # ('2021-01-01', '2022-01-01', '2022-07-01', 'RB'), \
        # ('2020-01-01', '2022-01-01', '2022-07-01', 'RB'), \
]
result_df = pd.DataFrame()
for start_train_date, end_train_date, end_test_date, target_inv in target_inv_list:
    filename = f'result/gplearn_{target_inv}_1min_factors{start_train_date}_{end_train_date}_{end_test_date}.pkl'
    with open(filename, 'rb') as f:
        est_gp = cloudpickle.load(f)
        '''
        读取最优因子
        '''
        best_programs = est_gp._best_programs
        best_programs_dict = {}

        for p in best_programs:
            factor_name = 'alpha_' + str(best_programs.index(p) + 1)
            best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
            
        best_programs_dict = pd.DataFrame(best_programs_dict).T
        best_programs_dict = best_programs_dict.sort_values(by='fitness')
        '''
        读取原始数据
        '''
        
        data_df = pd.read_hdf(f'dataset/{target_inv}.h5')
        data_df = data_df[start_train_date:end_test_date]
        fields = ['open','close','high','low','volume','money']
        length = []

        df = data_df.copy()
        # 划分出训练集
        train_data=df[:end_train_date]
        train_data=train_data.reset_index(drop=True)

        test_data=df[start_test_date:]
        test_data=test_data.reset_index(drop=True)

        scaler = MinMaxScaler()
        X_train = train_data.drop(columns=['date','1_min_return']).to_numpy()
        X_train = scaler.fit_transform(X_train)
        y_train = train_data['1_min_return'].values
        X_train, X_train.shape, y_train, y_train.shape
        X_test = test_data.drop(columns=['date', '1_min_return']).to_numpy()
        X_test = scaler.transform(X_test)
        y_test = test_data['1_min_return'].values
        X_test, X_test.shape, y_test, y_test.shape
        # for i in range(10):
        #     graph1 = alpha_factor_graph(i, best_programs_dict)
        #     graph1.render(f'images/{target_inv}_1min_factors{start_train_date}_{end_train_date}_{end_test_date}_alpha_{i}', format='png', cleanup=True)
        
        pd.DataFrame(best_programs_dict).to_excel(f'images/{target_inv}_1min_factors{start_train_date}_{end_train_date}_{end_test_date}_factor.xlsx')

        factors_pred_1 = est_gp.transform(X_test)
        factors_pred_1.shape, X_test.shape
        # 去掉异常的因子
        pred_data_1 = pd.DataFrame(factors_pred_1).astype(float)
        pred_data_1, pred_data_1.iloc[:,[0]]
        drop_index = []
        for col in pred_data_1.columns:
            if list(pred_data_1[col]).count(0) > 0.5 * pred_data_1.shape[0]:
                drop_index.append(col)
        pred_data_1.drop(drop_index, axis=1, inplace=True)
        # test_data = test_data.astype(float)
        test_data.money = test_data.money.astype(float)
        test_data.volume = test_data.volume.astype(float)
        total_return = _factor_backtest(pred_data_1.iloc[:,[0]].values, test_data)
        total_return = total_return.pct_change(1).fillna(0)
        # total_return[total_return==float('inf')] = 0
        # total_return[total_return==float('-inf')] = 0
        # total_return.fillna(0, inplace=True)
        result_df[f'{target_inv}_{start_train_date}_{end_train_date}_{end_test_date}'] = total_return
result_df.to_excel('images/result_ret.xlsx')
