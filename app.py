import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import os
import hashlib

# 安全导入 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. 配置持久化管理 (Config Persistence)
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# 默认标的池
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'persistence_days': 3,    # [New] 必须连续第一的天数
    'vol_filter_window': 20,  # [New] 成交量均线周期
    'vol_min_ratio': 0.6,     # [New] 最小成交量占比(当日/均量)
    'allow_cash': True,
    'mom_method': 'Risk-Adjusted (稳健)', 
    'selected_codes': DEFAULT_CODES
}

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except Exception as e:
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# ==========================================
# 1. 投行级页面配置 & CSS样式
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 量价增强版",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #eaeaea; border-radius: 12px;
        padding: 20px 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center; height: 100%; transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.08); }
    .metric-label { color: #7f8c8d; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #2c3e50; font-size: 1.6rem; font-weight: 700; }
    .metric-sub { font-size: 0.8rem; color: #95a5a6; margin-top: 6px; }
    .signal-banner {
        padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
    }
    .dataframe { font-size: 13px !important; border: 1px solid #eee; }
    .opt-highlight { background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001 
PRESET_ETFS = {
    "518880": "黄金ETF (避险)", "588000": "科创50 (硬科技)", "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)", "159915": "创业板指 (成长)", "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)", "512890": "红利低波 (防御)", "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)", "512880": "证券ETF (Beta)"
}

def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

# ==========================================
# 2. 数据层 (Data Layer) - 增加成交量获取
# ==========================================

@st.cache_data(ttl=3600*12) 
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list, end_date_str):
    """
    下载历史数据：收盘价 + 成交量
    """
    start_str = '20150101' 
    price_dict = {}
    vol_dict = {}
    name_map = {}
    
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty:
                name = match.iloc[0]['名称']
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                price_dict[name] = df['收盘'].astype(float)
                vol_dict[name] = df['成交量'].astype(float)
        except Exception:
            continue

    if not price_dict:
        return None, None, None

    # 合并并清洗数据
    df_price = pd.concat(price_dict, axis=1).sort_index().ffill()
    df_vol = pd.concat(vol_dict, axis=1).sort_index().fillna(0)
    
    df_price.dropna(how='all', inplace=True)
    # 对齐索引
    common_idx = df_price.index.intersection(df_vol.index)
    df_price = df_price.loc[common_idx]
    df_vol = df_vol.loc[common_idx]

    if len(df_price) < 20: return None, None, None
    return df_price, df_vol, name_map

# ==========================================
# 3. 策略内核 (Strategy Core) - 增强版
# ==========================================

def calculate_momentum(price_df, vol_df, lookback, smooth, method, 
                      vol_filter_active=False, vol_window=20, vol_min_ratio=0.6):
    """
    计算动量，可选加入成交量过滤
    """
    # 1. 基础动量计算
    if method == 'Classic (普通)':
        mom = price_df.pct_change(lookback)
    elif method == 'Risk-Adjusted (稳健)':
        ret = price_df.pct_change(lookback)
        vol = price_df.pct_change().rolling(lookback).std()
        mom = ret / (vol + 1e-9)
    elif method == 'MA Distance (趋势)':
        ma = price_df.rolling(lookback).mean()
        mom = (price_df / ma) - 1
    else:
        mom = price_df.pct_change(lookback)

    if smooth > 1:
        mom = mom.rolling(smooth).mean()

    # 2. 成交量过滤逻辑 (Volume Filter)
    # 如果当日成交量 < 过去N天均量 * ratio，则认为上涨无力或无效，将动量置为负无穷或惩罚
    if vol_filter_active and vol_df is not None:
        vol_ma = vol_df.rolling(vol_window).mean()
        # 避免除以0
        vol_ratio = vol_df / (vol_ma + 1e-9)
        
        # 创建掩码：如果 vol_ratio < threshold，则掩盖动量
        # 注意：这里我们只惩罚“低量”状态，不奖励“放量”状态，因为ETF放量有时也是顶
        low_vol_mask = vol_ratio < vol_min_ratio
        
        # 将低量日的动量设为极小值，使其在排名中垫底
        mom = mom.mask(low_vol_mask, -1.0) # 或者 np.nan, 但 -1.0 更能保证不做多

    return mom

def robust_backtest(daily_ret, mom_df, threshold, min_holding=3, 
                   persistence_days=3, cost_rate=0.0001, allow_cash=True):
    """
    增强版回测引擎：加入 Persistence (持续天数) 逻辑
    """
    signal_mom = mom_df.shift(1) # 昨日动量决定今日操作
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    
    # 状态变量
    curr_idx = -2    # -2:未建仓, -1:空仓(Cash), >=0:持仓资产索引
    days_held = 0    # 当前资产持有天数
    trade_count = 0
    
    # [New] 潜在候选者逻辑
    candidate_idx = -2
    candidate_days = 0 
    
    # 记录历史持仓以便绘图
    holdings_log = [-2] * n_days 

    for i in range(n_days):
        # 记录持有天数
        if curr_idx != -2:
            days_held += 1
            
        row_mom = p_mom[i]
        
        # 如果整行数据无效（通常是开始几天），跳过
        if np.isnan(row_mom).all(): 
            holdings_log[i] = curr_idx
            continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        
        # 1. 找出今日理论最强
        today_best_idx = np.argmax(clean_mom)
        today_best_val = clean_mom[today_best_idx]
        
        # 2. 处理 [榜首持续性] 逻辑 (Persistence)
        # 只有当同一个标的连续 N 天都是第一名，才被视为有效候选 (Valid Target)
        target_idx_final = curr_idx # 默认为保持现状

        # A. 绝对动量检查 (Cash Check)
        market_is_bad = False
        if allow_cash and today_best_val < 0:
            market_is_bad = True
        
        if market_is_bad:
            # 市场不好，直接考虑切空仓，不需要 Persistence (逃跑要快)
            target_idx_final = -1
            # 重置候选状态
            candidate_idx = -1
            candidate_days = 0
        else:
            # B. 相对动量检查
            # 更新候选者计数器
            if today_best_idx == candidate_idx:
                candidate_days += 1
            else:
                candidate_idx = today_best_idx
                candidate_days = 1
            
            # C. 判断是否满足切换条件
            # 条件1: 候选者连续第一的时间 >= persistence_days
            is_candidate_solid = (candidate_days >= persistence_days)
            
            # 逻辑分支
            if curr_idx == -2:
                # 初始建仓：只要有有效数据且大于0即可，稍微宽松一点，或者也要求 persistence
                if today_best_val > -np.inf: target_idx_final = today_best_idx
            
            elif curr_idx == -1:
                # 从空仓抄底：必须满足 persistence，防止骗线
                if is_candidate_solid: target_idx_final = candidate_idx
                
            else: # 当前有持仓
                # 只有当 1.持仓满足最小天数 AND 2.新候选者地位稳固 AND 3.优势超过阈值 时才换
                if days_held >= min_holding:
                    if is_candidate_solid and (candidate_idx != curr_idx):
                        curr_val = clean_mom[curr_idx]
                        cand_val = clean_mom[candidate_idx]
                        if cand_val > curr_val + threshold:
                            target_idx_final = candidate_idx
        
        # 3. 执行交易逻辑
        if target_idx_final != curr_idx:
            if curr_idx != -2: # 只要不是第一次建仓，就算交易
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            curr_idx = target_idx_final
            
        # 4. 计算当日收益
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
        
        holdings_log[i] = curr_idx
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    return total_ret, max_dd, equity_curve, trade_count, holdings_log

# ==========================================
# 4. 分析师工具箱 (优化函数)
# ==========================================

def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):
    if len(equity_curve) < 2: return {}
    s_eq = pd.Series(equity_curve)
    daily_ret = s_eq.pct_change().fillna(0)
    days = len(equity_curve)
    
    total_ret = equity_curve[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    return {
        "Total Return": total_ret, "CAGR": ann_ret, "Volatility": ann_vol,
        "Max Drawdown": max_dd, "Sharpe Ratio": sharpe, "Trades": trade_count
    }

def optimize_parameters(price_df, vol_df, allow_cash, min_holding, persistence_days, vol_filter_active):
    """
    参数优化：主要优化 周期、平滑、阈值
    """
    methods = ['Risk-Adjusted (稳健)', 'MA Distance (趋势)']
    lookbacks = [20, 25, 30] 
    smooths = [3, 5]      
    thresholds = [0.001, 0.005, 0.010]
    
    daily_ret = price_df.pct_change().fillna(0)
    n_days = len(daily_ret) 
    results = []
    
    total_iters = len(methods) * len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text="正在进行参数扫描...")
    
    idx = 0
    for method in methods:
        for lb in lookbacks:
            for sm in smooths:
                # 预计算动量
                mom = calculate_momentum(price_df, vol_df, lb, sm, method, 
                                       vol_filter_active=vol_filter_active)
                for th in thresholds:
                    ret, dd, equity, count, _ = robust_backtest(
                        daily_ret, mom, th, 
                        min_holding=min_holding,
                        persistence_days=persistence_days,
                        cost_rate=TRANSACTION_COST, 
                        allow_cash=allow_cash
                    )
                    
                    ann_ret = (1 + ret) ** (252 / n_days) - 1
                    sharpe = 0
                    if n_days > 1:
                        eq_s = pd.Series(equity)
                        d_r = eq_s.pct_change().fillna(0)
                        ann_vol = d_r.std() * np.sqrt(252)
                        sharpe = (ann_ret - 0.03) / (ann_vol + 1e-9)
                    
                    ann_trades = count * (252 / n_days)
                    results.append([method, lb, sm, th, ret, ann_ret, count, ann_trades, dd, sharpe])
                    
                    idx += 1
                    my_bar.progress(min(idx / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['方法', '周期', '平滑', '阈值', '累计收益', '年化收益', '调仓次数', '年化调仓', '最大回撤', '夏普比率'])
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        saved_config = load_config()
        st.session_state.params = saved_config

    with st.sidebar:
        st.title("🎛️ 策略控制台 (Pro)")
        
        # --- 1. 资产与数据 ---
        st.subheader("1. 资产池配置")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        
        default_display = []
        if not all_etfs.empty:
            for code in current_selection_codes:
                match = all_etfs[all_etfs['代码'] == code]
                if not match.empty:
                    default_display.append(match.iloc[0]['display'])
                else:
                    # 如果找不到（比如代码输入错误），保持原样或跳过
                    pass
        
        # 修复多选框默认值逻辑
        final_defaults = [x for x in default_display if x in options]
        if not final_defaults and current_selection_codes: # 如果匹配失败但有代码，尝试直接用代码
             pass 

        selected_display = st.multiselect("核心标的池", options, default=final_defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        st.subheader("2. 回测区间")
        start_date = st.date_input("开始日期", datetime(2021, 1, 1))
        end_date = st.date_input("结束日期", datetime.now())
        initial_capital = st.number_input("初始资金", value=100000.0)

        st.divider()
        
        # --- 3. 策略参数 ---
        with st.form(key='settings_form'):
            st.subheader("3. 策略内核参数")
            
            mom_options = ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)']
            default_mom = st.session_state.params.get('mom_method', 'Risk-Adjusted (稳健)')
            p_method = st.selectbox("动量计算逻辑", mom_options, index=mom_options.index(default_mom) if default_mom in mom_options else 0)
            
            c1, c2 = st.columns(2)
            with c1:
                p_lookback = st.number_input("动量周期 (Days)", 10, 120, st.session_state.params.get('lookback', 25))
            with c2:
                p_smooth = st.number_input("平滑窗口 (Days)", 1, 20, st.session_state.params.get('smooth', 3))
                
            p_threshold = st.number_input("换仓阈值 (Buffer)", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
            
            st.markdown("#### 🛡️ 防抖动与风控 (核心增强)")
            
            c3, c4 = st.columns(2)
            with c3:
                p_min_holding = st.number_input("最小持仓天数", 1, 60, st.session_state.params.get('min_holding', 3), help="买入后至少持有几天才允许卖出")
            with c4:
                # [New] Persistence
                p_persistence = st.number_input("榜首确认天数", 1, 10, st.session_state.params.get('persistence_days', 3), help="必须连续N天排名第一才触发调仓信号")
            
            st.markdown("#### 📊 量价确认 (Volume Filter)")
            use_vol_filter = st.checkbox("启用缩量过滤 (Volume Check)", value=True)
            p_vol_window = st.slider("均量周期", 5, 60, st.session_state.params.get('vol_filter_window', 20))
            p_vol_ratio = st.slider("最低量比 (当日/均量)", 0.1, 1.0, st.session_state.params.get('vol_min_ratio', 0.6), help="如果当日成交量低于均量的这个比例，则视为动量无效")
            
            p_allow_cash = st.checkbox("启用绝对动量避险 (空仓)", value=st.session_state.params.get('allow_cash', True))
            
            submit_btn = st.form_submit_button("🚀 运行增强策略")

        if submit_btn:
            current_params = {
                'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
                'min_holding': p_min_holding, 'persistence_days': p_persistence,
                'vol_filter_window': p_vol_window, 'vol_min_ratio': p_vol_ratio,
                'allow_cash': p_allow_cash, 'selected_codes': selected_codes,
                'mom_method': p_method 
            }
            st.session_state.params = current_params
            save_config(current_params)

    # ================= 逻辑执行 =================
    st.title("🛡️ AlphaTarget | 量价增强策略终端")
    
    if not selected_codes:
        st.warning("请在左侧选择标的。")
        st.stop()
        
    with st.spinner("正在下载并清洗数据 (Price + Volume)..."):
        start_d = datetime.combine(start_date, datetime.min.time())
        end_d = datetime.combine(end_date, datetime.min.time())
        # 下载全量历史
        price_data, vol_data, name_map = download_market_data(selected_codes, end_d.strftime('%Y%m%d'))
        
    if price_data is None:
        st.error("数据获取失败，请检查网络或代码。")
        st.stop()

    # 切片
    mask = (price_data.index >= start_d) & (price_data.index <= end_d)
    sliced_price = price_data.loc[mask]
    sliced_vol = vol_data.loc[mask] if vol_data is not None else None
    
    if sliced_price.empty:
        st.error("选定区间内无数据。")
        st.stop()

    # 1. 计算动量 (含成交量过滤)
    mom_all = calculate_momentum(price_data, vol_data, p_lookback, p_smooth, p_method,
                                vol_filter_active=use_vol_filter, 
                                vol_window=p_vol_window, 
                                vol_min_ratio=p_vol_ratio)
    
    sliced_mom = mom_all.loc[mask]
    sliced_ret = sliced_price.pct_change().fillna(0)

    # 2. 执行增强版回测 (含 Persistence)
    ret, max_dd, equity_curve, trade_count, holdings_log = robust_backtest(
        sliced_ret, sliced_mom, p_threshold,
        min_holding=p_min_holding,
        persistence_days=p_persistence, # 传入确认天数
        cost_rate=TRANSACTION_COST,
        allow_cash=p_allow_cash
    )

    # 3. 结果展示
    df_res = pd.DataFrame({
        '策略净值': equity_curve,
        '基准净值': (1 + sliced_ret.mean(axis=1)).cumprod()
    }, index=sliced_price.index)
    
    # 还原持仓名称
    holding_names = []
    code_list = sliced_price.columns.tolist()
    for idx in holdings_log:
        if idx == -2: holding_names.append("建仓中")
        elif idx == -1: holding_names.append("Cash (空仓)")
        else: holding_names.append(name_map.get(code_list[idx], code_list[idx]))
    df_res['持仓'] = holding_names
    
    # 计算指标
    metrics = calculate_pro_metrics(equity_curve, df_res['基准净值'].values, trade_count)
    
    # --- UI 组件 ---
    
    # A. 核心指标卡
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(metric_html("累计收益", f"{metrics.get('Total Return',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益", f"{metrics.get('CAGR',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("最大回撤", f"{metrics.get('Max Drawdown',0):.1%}", "", "#27ae60"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("夏普比率", f"{metrics.get('Sharpe Ratio',0):.2f}", "", "#2c3e50"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("调仓次数", f"{trade_count}", f"年化: {trade_count * (252/len(sliced_price)):.1f}", "#2c3e50"), unsafe_allow_html=True)

    # B. 信号横幅
    last_h = holding_names[-1]
    st.markdown(f"""
    <div class="signal-banner">
        <h3 style="margin:0">📌 当前持仓: {last_h}</h3>
        <div style="margin-top:5px; font-size: 0.9rem">
            风控状态: 榜首确认 {p_persistence} 天 | 最小持有 {p_min_holding} 天 | 缩量过滤: {"ON" if use_vol_filter else "OFF"}
        </div>
    </div>""", unsafe_allow_html=True)

    # C. 图表区域
    tab1, tab2 = st.tabs(["📈 净值与持仓", "🛠️ 参数优化"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # 净值
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['策略净值'], name="策略净值", line=dict(color='#d63031', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['基准净值'], name="等权基准", line=dict(color='#b2bec3', dash='dash')), row=1, col=1)
        
        # 持仓色块 (用散点图模拟甘特图效果，或者直接画在副图)
        # 为了直观，我们在副图画出持有资产的类别代码
        # 将持仓映射为数字以便绘图
        unique_holds = list(set(holding_names))
        hold_map_y = {name: i for i, name in enumerate(unique_holds)}
        y_vals = [hold_map_y[h] for h in holding_names]
        
        fig.add_trace(go.Scatter(
            x=df_res.index, y=y_vals, mode='markers', 
            marker=dict(size=5, color=y_vals, colorscale='Viridis'),
            name="持仓分布", showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            height=600, 
            hovermode="x unified",
            yaxis2=dict(tickmode='array', tickvals=list(hold_map_y.values()), ticktext=list(hold_map_y.keys()))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示近期交易记录
        st.subheader("📝 近期持仓明细")
        st.dataframe(df_res.tail(20).style.format({'策略净值': '{:.4f}', '基准净值': '{:.4f}'}))

    with tab2:
        st.info("提示：此优化将基于当前选择的标的和时间段，寻找最佳的 [动量周期] 和 [换仓阈值]。")
        if st.button("开始参数扫描"):
            opt_df = optimize_parameters(sliced_price, sliced_vol, p_allow_cash, p_min_holding, p_persistence, use_vol_filter)
            
            best_sharpe = opt_df.loc[opt_df['夏普比率'].idxmax()]
            best_ret = opt_df.loc[opt_df['累计收益'].idxmax()]
            
            c_o1, c_o2 = st.columns(2)
            with c_o1:
                st.markdown('<div class="opt-highlight">💎 <b>夏普最优</b></div>', unsafe_allow_html=True)
                st.write(f"配置: {best_sharpe['方法']} | 周期: {best_sharpe['周期']} | 阈值: {best_sharpe['阈值']}")
                st.write(f"夏普: {best_sharpe['夏普比率']:.2f} | 年化: {best_sharpe['年化收益']:.1%}")
            
            with c_o2:
                st.markdown('<div class="opt-highlight">🔥 <b>收益最优</b></div>', unsafe_allow_html=True)
                st.write(f"配置: {best_ret['方法']} | 周期: {best_ret['周期']} | 阈值: {best_ret['阈值']}")
                st.write(f"夏普: {best_ret['夏普比率']:.2f} | 年化: {best_ret['年化收益']:.1%}")
                
            st.dataframe(opt_df.sort_values('夏普比率', ascending=False).head(10))

if __name__ == "__main__":
    main()
