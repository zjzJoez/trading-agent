# 交易系统复活方案(FINAL)

**日期**: 2026-07-20 · **基线**: main @ edefe30 (2026-06-12 部署) · **版本**: FINAL(经 statistical-rigor / deployability / edge-reality 三方对抗审查后修订)

**本版与草案的核心差异(先读这段)**:三份审查一致确认了草案的三个致命自伤:(1) 新评估门 `n=60 LB95<0` 以 70-95% 概率误杀一个完全按声明表现的策略——重蹈它自己诊断的 B3;(2) sleeve 1 的入场 gate 组合(delta 0.20-0.30 + credit ≥ width/3)在被允许交易的低 IV regime 里机械性不可满足——会复刻零订单漏斗;(3) 3-5 笔/周的节奏在 `MAX_CONCURRENT_OPENS = 6`(`sizing.py:66`,已在代码确认)下算术上不可能,且 combo 的平仓路径完全不存在(`exits/` 与 `intraday_nodes.py` 零 combo 处理,已确认)——sleeve 1 的全部 edge 来自管理规则,而管理规则目前无法执行。本版逐条修复,时间表按真实约束重算:**n=30 健康检查 ≈ 第 4 个月,首次 kill-gate 检查 ≈ 第 6-7 个月**,不再声称 8-12 周。

---

## 第一部分:诊断(与草案基本一致,一处修订)

### A 类:系统瞎了/坏了

**A1. rvol bug —— 零订单直接根因。** `premarket_nodes.py:193-194` 用当日盘中累计成交量除以 `avg_volume_3m`,注释却声称是 prior-day volume;唯一执行扫描在 9:35 ET(开盘 5 分钟),rvol 机械读出 0.1-0.7x,prompt 教 scout "≥1.5x 才算活跃",scout 甚至幻觉 "still pre-open"。
【修订】草案用 7/8-7/17 的 "8/8 天 12:30 top score ≥ 0.50" 同时证明 bug 存在和 "0.50 阈值没病"——后者是 8 天 in-sample 证据的二次使用(8/8 与任何真实通过率 ≥69% 都相容,且该窗口 CRNX 独占 6/8 天)。**结论降级**:反事实只证明 rvol 压制了 dispatch;0.50 阈值去留改为修复后 10 个交易日数据上的预注册决策(见 Week 1 Step 6)。

**A2. 入场只占交易周 ~2% 且钉死在 9:35 ET** —— 违反操作者四月复盘规则 #6(9:30-9:45 禁止交易)。入场瞎、退出灵,结构性不对称。
【修订新增】现有 timer 是固定 UTC:13:35 UTC 只在 EDT 期间是 9:35 ET,11 月冬令时后变 8:35 ET(盘前)→ `is_us_market_open()` 让所有执行扫描降级 digest-only → **静默回到零订单**。这是现网潜伏 bug,新 timer 必须按 timezone 写(见 Step 5)。

**A3. OAuth 静默死机 18 天,报警无升级路径**(471 次 `trade_engine_silence_detected` 无人响应;NVDA -5.93R 占账本净亏 52%)。

**A4. 已建成零部署的能力**:jennings 分支(EDGAR async bug——scout 从未见过 filings 文本;对账;oneshot 修复)、lehmann 分支(R5e credit verticals、regime 减仓 P0b、thesis-broken 退出 P0c、learning 接线,645/645 测试绿)。
【修订新增,edge-reality 确认】lehmann 的 combo 是**只进不出**:`place_paper_option_combo` 只有开仓;combo 在 journal 里是一行、以 short 腿 OCC symbol 为键、entry_price = net credit;现有退出引擎会拿 short 腿单腿价格对比 net credit(P&L 错误),任何触发只平 short 腿、**遗孤 long 腿**。lehmann 自己文档标注 "no first-class combo-CLOSE path"。

**A5. Gate 体系不自洽**(spec bands 无处强制、R6 死代码、`approved_qty=0` 静默、decline 后无 cooldown、R5d 5% < 实测中位点差 7.3%)。

### B 类:没有 edge 可表达

**B1** 单腿 long premium + 7.3% 点差 = 文献级散户输家配置(Bryzgalova et al. 2023);障碍期权几何学下毛 EV 恒为零,每笔背 0.18-0.30R 摩擦。**B2** 止损设在 ~1.0 日 σ,2 日触碰概率 43-48%。**B3** `n=30 LB95<0` 门对按声明 mid-band 表现的策略误杀率 88%,证实需 n≈367。**B4** LLM 在给数字表排序(无价值处),EDGAR bug 使它拿不到文本(有价值处)。**B5** learning 基础设施在当前成交量下数学上无结论可产出。

---

## 第二部分:WEEK 1 —— 解封

【修订】重命名为 **"Week 1 部署,Week 2 验收"**:Step 4/6 的验收各需 5-10 个交易日配对观察,压缩成 7 天只会诱发跳过验证。另注:**auto-deploy 不做测试门禁**(每 2 分钟轮询 origin/main 直接部署),Step 1 与 Step 3 之间 main 带着 7 个过期日期测试失败运行——可接受,但在此言明。

**Step 1. Merge `claude/strange-jennings-53491a` → main,重跑 `install_timers.sh`。**
验收:premarket payload 出现 filing headline(无 `coroutine never awaited` warning);`run_finished.nodes_visited` 非空;首次 22:00 UTC reconcile 跑绿。

**Step 2. EC2 Postgres 先执行 migration 014**(必须在 lehmann push 前,否则缺列使 `get_open_journal_trades` fail-closed → 退出引擎降级 HOLD)。
【修订验收,deployability-minor】草案验收("EC2 上正常返回")对已部署代码空洞成立——main 版本根本不 SELECT 新列。改为:列存在于 EC2 Postgres,**且**从 lehmann checkout 用 smoke 脚本连 EC2 DB 跑 `get_open_journal_trades` 绿。

**Step 3. Merge `claude/great-lehmann-18cf01` + `compare_positions` combo 腿展开补丁**(用 `market_snapshot` 的双腿 symbol 展开 combo 行,防每晚假 `position_without_journal_row` 告警)。
【修订验收,三方一致】**删除"开一笔测试 vertical"验收** —— 已部署的退出引擎会错管这个仓位(A4 新段)。改为:全测试套件绿;`credit_put_spread_30_45` status=active;`_run_combo_guard` dry-run(不成交)通过;reconcile 补丁用**模拟 snapshot** 验证无假警。真实 vertical 首开推迟到 Month 1 combo-close 路径落地后。

**Step 4. 修 rvol —— time-of-day 基线,不是均匀 elapsed 归一化。**
【修订,deployability-major】草案的 `today_vol / (avg_3m × elapsed)` 忽略 U 形日内成交量分布:10:15 ET 时 elapsed = 45/390 ≈ 0.115,但正常累计量已是全天 20-30% → 每个普通名字读出 2-2.5x,配合 prompt 的 "≥1.5x" 教条,故障从"全压制"翻转成"全报警"。正确基线是**同时刻累计量**:
```
v1 (cheap):   rvol = today_cum_vol(t) / (avg_volume_3m * f(t))
              # f(t) = 静态市场级累计成交量曲线(U 形 profile)
v2 (per-ticker): rvol = today_cum_vol(t) / avg_trailing_20d(cum_vol_by_time(t))
              # 由 moomoo get_historical_kline 日内 bar 计算(工具已存在)
盘前(12:30 digest): rvol = prior_day_volume / avg_volume_3m(维持现语义)
```
同一 commit 内:修正错误注释;prompt 传入当前时间与开盘经过分钟数;**重校 prompt 阈值**("≥1.5x" 是按 prior-day 比率校准的,对 time-normalized 比率不自动成立)。
【修订验收,可判定化】草案的"落在 12:30 值合理带内"比较的是两个不同量纲、无带定义,不可判定。改为:连续 5 个交易日,每次执行扫描 watchlist 全体的 **median rvol ∈ [0.7, 1.4]**(无偏估计量应中心于 1.0);`candidates_ranked` 理由零 "pre-open" 字串。

**Step 5. 执行窗口 + decline cooldown 同批落地:10:15 ET 与 13:30 ET 双窗口,timer 按 timezone 写。**
【修订,deployability-minor×2】(a) timer 写成 `OnCalendar=Mon..Fri *-*-* 10:15:00 America/New_York`(systemd 支持 tz 后缀),否则 11 月 DST 切换后静默回退零订单——现网 13:35 UTC timer 就带着这颗雷;(b) 草案把双窗口(旧 Step 5)排在 cooldown(旧 6a)之前,意味着同一 ticker 每天烧两遍完整 LLM 流水线——CRNX 病理×2。**decline/VETO 后同 ticker 3 交易日 cooldown 与双窗口同一次部署**。(c) 新增 healthcheck 断言:任何交易日零 dispatch-eligible 扫描 → ntfy 高优推送(顺带服务 Step 7)。
验收:dispatch 事件出现在新时刻;被拒 ticker 3 日内零重 dispatch;模拟 tz 切换日 timer 仍解析到 10:15 ET。

**Step 6. Gate 微调。** (a) `approved_qty ≤ 0` → 专用事件 `order_skipped_zero_qty`(`trade_nodes.py:893-895` 现静默返回);(b) proposal schema 层强制 spec bands(`dte_range`/`abs_delta_range`),坏提案进流水线前死掉——回放 7/8 CRNX delta-0.815 提案必须在 schema 层被拒;(c)【修订】0.50 阈值:**预注册决策规则取代 in-sample 结论**——rvol 修复后收集 10 个交易日配对日志,若 dispatch 率落在 2-5 次/周则保留 0.50,否则按新数据调整。规则先写下,数据后到。

**Step 7. Deadman 升级 + 每日熔断。** 市场时段静默 >24h → ntfy 高优推送(非 Telegram,5/28 已移除);>48h → 自动 halt 新开仓(不自动平仓——故障可能在报警侧);当日已实现亏损 ≥ 2R → 当日停开新仓。
【修订,edge-reality-minor】deadman 链路验证通过是 **sleeve 1 首笔真实开仓**的前置条件,不只是扩规模的前置——一个静默死过 18 天的系统在报警链路修好前不配持有任何短 premium。
验收:模拟静默触发推送与 halt;模拟 -2R 触发熔断事件。

**Week 1-2 明确不做**:不动 HMM、不动 learning 内部、不碰 frozen holdout、不开任何真实 vertical。

---

## 第三部分:MONTH 1 —— Edge 组合

【修订总纲】三处结构性改动:(1) **combo-close 路径升格为第一位、阻塞性工作流**——它不是"跟进项",它是 sleeve 1 全部声明 edge(管理规则)的执行前提;(2) **sleeve 2 (insider) 整体移到 Month 2**——它的扫描 job 需要新增每 insider 3-5 年 Form-4 历史抓取(LLM 读单份 filing 无法计算 calendar regularity),Month 1 塞不下;(3) **`earnings_dte` 接入延期到 Quarter**——指数 ETF 无 earnings,Month 1 无任何消费者,mega-cap 扩展本来就 gated 在它上面,天然的缓冲项。

### M1-0. Sleeve 1 首笔开仓的阻塞性前提(全部完成才准第一笔 fill)

1. **Combo-aware 退出路径**(deployability/edge-reality 双 critical):net-of-both-legs 标记(替换 `refresh_quotes_and_greeks` 的单 symbol last_price 逻辑)、原子双腿平仓工具(平仓先平 short 腿,带 rollback,与开仓对称)、`hard_executor` 识别 combo 行(从 `market_snapshot` 展开双腿、按 spread 整体定价 50%-PT 与 21-DTE 触发)、定义 P0b "trim 50%" 对 1-2 张 vertical 的语义(整 unit 平仓或跳过,不拆腿)。
   验收:开一笔测试 vertical → 监控标记 net-of-legs 价值 → 50%-PT 与 21-DTE 触发各自**平掉双腿**,journal 验证;当日手动清理。
2. **Combo cost-honest paper fill**(edge-reality-minor 升格):双腿双向按 `execution_costs.py` 收费,一笔完整 round-trip 验证后才准计入任何评估样本——0.03-0.05R 的声明 margin 对 3-4pp 的 WR 缓冲是决定性的,被美化的早期 fill 会污染 n=30。
3. **Per-underlying 点差校准**(edge-reality-minor):用现有 `execution_costs.py` 校准机器对准 SPY/QQQ/IWM 三条链采集 live quotes,发布 per-underlying `friction_r` 进 spec;**IWM 实测 combo friction > 0.06R 则移出白名单**——1-3% 点差是 SPY/QQQ 级假设,IWM 低权利金 long wing 常见 2-5%+/腿,不让 per-leg 5% gate 仲裁。
4. **Gate 可行性回放 + managed-payoff 期望重算**(三方 critical,快照回测器从 Quarter 提前):对 `option_chain_snapshots`(6/12 起每日累积)回放完整 gate 栈,统计每 underlying 每日合格 vertical 数、分 regime 标签;同时从 21-DTE 快照 marks 估计管理规则下的真实盈亏分布。验收:**允许交易的 regime 内合格 vertical 存在于 ≥60% 快照日**,否则调 gate 并重算 breakeven,不部署第二个带更好文档的零订单漏斗。

### M1-1. `credit_vertical_index_30_45` spec(修订版)

```python
StrategySpec(
    name="credit_vertical_index_30_45",
    status="pending_prereqs",          # M1-0 全绿才转 active
    structure="Short put vertical (BULL/RANGE); short call vertical (BEAR, regime 确认时). "
              "30-45 DTE, short leg |delta| 0.20-0.35, width $5-10, "
              "atomic combo via place_paper_option_combo",
    entry_gates={
        "underlying_whitelist": ("SPY", "QQQ"),        # IWM 待 M1-0.3 实测后定
        "dte_range": (30, 45),
        "abs_delta_range": (0.20, 0.35),               # 【修订】上限 0.30→0.35
        "min_credit_frac_of_width": 0.25,              # 【修订】width/3 → width/4 (占位)
        # 【修订】删除 min_risk_reward=0.40 —— 与 credit gate 数学冗余且草案版互相矛盾
        "max_spread_pct_mid": 0.05,
        "news_veto_required": True,                    # 单次 LLM 调用
    },
    allowed_regimes=("BULL_TREND", "RANGE_LOW_VOL"),   # VOL_TRANS/BEAR 减半, CRISIS=0
    # 管理规则(deterministic, hard_executor): 50% profit-take 或 21-DTE 强平; 无中途止损
    expectancy_profile="M1-0.4 快照回放产出的 managed-payoff 分布,不再用 expiry-binary 公式",
    ...
)
```

【修订说明,三方 critical】(a) 草案 `credit ≥ width/3` 要求 credit/width ≥ 0.33,而 0.20-0.30 delta 的 30-45 DTE 指数 put spread 正常 IV 下只收 ~0.18-0.25——≥0.33 只在高 IV 出现,而高 IV 映射到被 spec 减半/清零的 regimes:**结构性复刻零订单**。占位改为 `credit ≥ width/4`(gross breakeven 75%)+ delta 上限放宽到 0.35,**最终值由 M1-0.4 回放决定**。(b) 草案的 `breakeven_wr_gross(0.40)=71.4%` 假设 hold-to-expiry binary payoff,与自己的 50%-PT 管理规则矛盾(win 封顶 ~0.5×credit 把同模型 breakeven 推到 ~83%,而 21-DTE 强平又截断亏损)——两个方向都没算。期望画像必须建在 managed-payoff 分布上,数据来源就是快照回放。(c)【edge-reality-major,接受】**PUT index Sharpe 0.65 引用撤回**:那是 cash-secured naked SPX put,买回远 OTM wing(vol surface 最贵区域)会交还不成比例的 VRP;defined-risk vertical 的文献/实证(spintwig 类回测)是净接近零到温和为正。Spec 文本明写 wing-cost haircut,预期净 Sharpe 声明降为 0.2-0.4,且现实 EV 带**跨零**——正因如此,benchmark domination test(见 Quarter)是预注册的诚实出口。

### M1-2. Per-sleeve 并发预算(需操作员批准,见 Ask #6)

【修订,三方 critical/major】现行 R2 `MAX_CONCURRENT_OPENS = 6`(股票+期权合计)下,sleeve 1 的 3-5 开/周 × 2-3 周持有 = 7.5-15 并发需求,叠加 sleeve 2 的 4-18 并发——需求是上限的 2-8 倍,且 `order_guard` 在每次下单重验 R1-R7,绕过 dispatch 流水线也绕不过 cap。同时 `SAME_TICKER_COOLDOWN_DAYS = 7`(`premarket_nodes.py:504`)+ same-underlying exposure gate 在 3 名单 universe 上恰好禁止每周阶梯建仓。改:
- **R2a**:defined-risk 指数 vertical ≤ 9 并发,**且 aggregate max_loss ≤ 15% equity**(有界 max_loss 是提额的正当性来源;两约束取先绑者);
- **R2b**:股票 sleeve ≤ 4 并发;**R2c**:其余 ≤ 2;
- 白名单 underlying 的 cooldown 与 same-underlying gate 改为 **per-expiry** 作用域(同一 expiry 不叠仓,不同 expiry 阶梯放行)。

**诚实节奏表(取代草案的 "n=30 at week 8-10")**:closes/week = 并发上限 ÷ 中位持有周数 = 9 ÷ 2.5 ≈ **3.6 平仓/周**(sleeve 1 独立预算,不与 sleeve 2 竞争)。首笔 fill 在 M1-0 完成后(≈ 第 5-6 周),则:**n=30 机械健康检查 ≈ 第 15-17 周(第 4 个月)**;**n=60 kill-gate 检查 ≈ 第 6-7 个月**。草案把 8-12 周的"机械健康检查"混称为 "LB95 判定"——两者是不同的门,此处分开。

### M1-3. 评估合同(修订版)+ T0 预注册

【修订,三方 critical——草案新门被证明 70-95% 误杀,重蹈 B3】三层结构:

1. **n=30 机械健康检查**(可以"通过"):(a) 实付成本 vs M1-0.3 **实测**校准值偏离 ≤ 1.5×(不再对着未验证的 0.03-0.05R 假设循环验证);(b) 无单笔亏损 > 1.05× 定义 max_loss;(c) 观测 WR vs **management-aware 基线**(P(50%-PT before 21-DTE | short delta),来自快照回放——不是 hold-to-expiry 的 1−delta,profit-taking 会机械抬高观测 WR,delta 基线的 null 从第一天就错)。
   【修订,stat-minor】功效诚实标注:binomial 检验 n=30 对真实 WR 65% 只有 ~37% power——它抓的是灾难性失准(≥20pp 缺口),不是 band 边缘欠佳;**(a)(b) 才是 n=30 门真正的牙齿**(逐笔近确定性),spec 里如此标注。
2. **n=60 kill-gate**:moratorium 条件改为 **LB95(mean R) < −0.10R**("被证明有害",不是"未被证明有利")。发布的 operating characteristics(spec 正文):真实 mean = +0.05R、σ ≈ 0.35R(managed payoff)时误杀率 ≈ **5%**(草案的 LB95<0 版本同参数下误杀 ≈ 70%)。CI 按 entry-week block-bootstrap——同周开的 SPY/QQQ vertical 相关性 >0.9,iid Student-t 在负偏尾部反保守。
3. **升级/扩规模是独立的门**:LB95(mean R) > 0 @ 97.5% 单侧,或跨两个不重叠窗口复现;诚实标注在声明经济学下需 n ≈ 150-400。**逐笔统计对 +0.05R 级 edge 在 n=60 就是统计上不可见的,因此账本级日度 benchmark window(vs 60/40,~60 obs/季,注明自相关)是 primary confirm/deny 统计量**——与 HMM validation 用的是同一框架。
4. **T0 预注册文档**(部署日,一份,不可事后改):每个 sleeve 的 checkpoint(固定 n,禁止中途 peek)、sleeve 3 复活标准及最小 n、mega-cap 扩展标准、benchmark domination 规则(见 Quarter)。多个 sleeve × 多个 95% 单侧门 × 复活/扩展通道 = 经典的 selection-by-multiplicity;family 一次写全,kill 用宽门、promote 用严门。

### M1-4. Sleeve 1 入场路径(枚举,不再隐含)

【修订,deployability-minor】规则化入场不是"删流水线"而是新建:scheduler job、基于 `get_option_chain` live greeks 的 strike selector、**thesis 自动记录**(`order_guard` 硬性要求下单前 10 分钟内有 thesis——草案漏了)、单次 news-veto LLM 调用、regime multiplier 在 trade_nodes 之外的消费 hook。合计 1-2 周工作量,列入预算。

### MONTH 2 —— `insider_cluster_stock`(从 Month 1 顺延)

【修订,edge-reality-major】(a) 声明 edge 从 CMP 82bps/月 **haircut 到 30-50bps/月**:1986-2007 估计 + 19 年 post-publication decay + 效应集中在被自家 ADV 下限排除的小盘——momentum 按 decay 打折了,insider 同规打折(草案的选择性怀疑,认账);(b) 扫描 job 定义补上**每个 clustered insider 抓取 trailing 3-5 年 Form-4 序列**,LLM 对着历史序列分类 routine(每年三月例行、10b5-1)vs opportunistic——读单份 filing 算不出 calendar regularity;(c) falsification 撤回草案的 "n=25 LB95<0"(同参数下 ~80% 误杀,与本方案自己的两级合同矛盾),纳入统一两级合同:机械健康检查 n=15-20(fill 质量、持有期合规、cluster 定义遵从),kill-gate 在误杀率 <30% 的 n(≈80-100,即**第二年**的事)。STOCK 执行、10bps 摩擦、R1 2.5%、4-12 周持有、R2b 预算内 1-2 笔/周。

### Sleeve 3: `convexity_long_premium` → shadow-only(零资金)

维持降级。【修订,stat-major】**复活标准重写**:方向 proxy(`evaluate_shadow_proposals.py` 的 20 日 underlying-direction)对一个死于摩擦/VRP/IV-crush 的策略什么都看不见,且每晚重复 look 无预注册 checkpoint 迟早随机打出 LB95>0——幸存者偏差复活通道。改为:复活只认 **option-level counterfactual P&L**(shadow proposals 对 `option_chain_snapshots` 回放 + `execution_costs.py` 收费,M1-0.4 的回放器使之几乎免费),最小 n 与 checkpoint 写进 T0 预注册,门用 promote 级严标准(97.5%)。方向 proxy 仅作日常监控仪表,永不作决策输入。

---

## 第四部分:QUARTER —— 复利循环 + 删除清单

- **快照回测器**:M1-0.4 已把最小版提前建成;Quarter 扩展为持续回放(仅 SPY/QQQ[/IWM] verticals、EOD 快照、`execution_costs.py` 收费)。验收:对已成交 paper fills 重放误差在成本模型容差内。
- **Benchmark domination test(预注册,新增)**:【edge-reality-major,接受】部署日 `start_benchmark_window.py` 注册 T0 vs 60/40 SPY/TLT;季度 checkpoint 若 **LB95(excess return vs 60/40) < 0 且逐笔健康检查全过** → 结论是"机器正常,edge 不存在"→ **wind down sleeve 1,不是调参**。60/40 是正偏度、零工时、零 outage 尾部风险的对照,它赢了就承认。
- **Sleeve 2 null test(重定性)**:【stat-major,接受】草案称"可证伪且几乎零成本"是统计不诚实:检出 ~55bps/20d 对 6-10% 的 20 日个股 idio vol 需每臂 ~700-2000 个 cluster,80% power;1-3 cluster/周一个季度只攒 12-36 个。重定性为**长期累积对照日志**:预注册最小 n ≥ 200 clusters(market-adjusted)后才准 read,**决策级答案在 1.5-3 年外**(如实告知操作员,Ask #4);近期预警靠 20 日 shadow-proxy 监控,并如此声明。
- **Learning/canary**:参数族收敛到新 sleeve 实际消费的(short-strike delta band、profit-take %、DTE band、`same_expiry_premium_max`);~3.6 平仓/周使 n=20 canary closes 首次在 1.5-2 个季度内可达;在此之前 learning loop 保持 idle。
- **`earnings_dte` 接入**(从 Month 1 顺延):mega-cap 扩展的前置;扩展决策本身走 T0 预注册的 promote 级门。
- **删除清单**(不变):`learning/archive.py` diversity archive(424 行);`regime/gates.py:44-156` 方向表(零调用者);LLM exit_monitor 死 schema;sleeve 1 入场不走 4-analyst+debate+Opus 流水线(每笔 5-8 分钟/7+ 次调用 → <1 分钟/1 次 news-veto);HMM v6 研究搁置;`docs/position_management.md` SELL 声明对齐 order guard;PARAM_BOUNDS 继续修剪。

---

## 第五部分:诚实性约束(系统明确不做什么)

1. **统计纪律**:kill 用 LB95(mean R) < −0.10R(宽门),promote 用 97.5% 或双窗复现(严门);账本级 benchmark window 是 primary confirm/deny;每个门在 spec 正文附**已计算的** operating characteristics(误杀率),不是"应附"。CI 按 entry-week block-bootstrap。
2. **成本诚实**:paper fills 按 `execution_costs.py` 收费;combo 双腿双向;7.3% 校准值用于 watchlist 域,指数域用 M1-0.3 **实测**值;成本检查对照实测基线,不对照假设(消除循环)。
3. **HMM**:只作 drawdown-control 开关,永不声称 alpha(IR −0.754 已判死);holdout ≥ 2026-06-13 冻结。
4. **永不裸卖**:单腿 SELL-to-open 硬封锁不解除;short 腿只存在于 R5e defined-risk combo 内。
5. **不做**:0DTE、overnight anomaly、真钱自动化(`enable_real` 冻结)、LLM 进退出热路径、LLM 中途调 stop/target、Telegram。
6. **负偏度与前置条件**:sleeve 1 最差月份在 vol spike;**deadman 链路验证 + combo-close 路径 + combo 成本模型三者全绿之前,零笔 sleeve 1 真实开仓**(不只是不扩规模)。
7. **诚实出口**:benchmark domination test 预注册——60/40 赢且机器健康 → wind down,不调参续命。

---

## 第六部分:Operator Asks(只有 Joez 能拍板的 6 个决定)

**1. 策略转向:convexity_long_premium 退役为 shadow-only,credit verticals(指数)转正为主力。** 建议:**批准**。文献级输家配置,全史 8 笔无一合规,旧评估门数学上打不出"通过"。复活通道保留但改为 option-level counterfactual(不是方向 proxy)。

**2. 执行窗口:9:35 ET 单窗口 → 10:15 ET + 13:30 ET 双窗口,timer 按 America/New_York 书写。** 建议:**批准**。9:35 违反你自己四月规则 #6;固定 UTC timer 在 11 月 DST 切换后会静默回到零订单——这是现网既有的雷,顺手拆掉。

**3. Deadman:ntfy 高优推送 + 48h auto-halt 新开仓(不自动平仓)+ 当日 −2R 熔断。** 建议:**批准**。【修订】此项从"扩规模前置"升格为 **sleeve 1 首笔开仓前置**。

**4. Sleeve 2 null test 重新决策。** 【修订,如实告知】它不是"一个季度出结果的便宜实验":在真实 cluster 流量下,决策级答案需要 ≥200 clusters ≈ **1.5-3 年**。请求批准的是**长期累积对照日志 + 预注册最小 n**,近期预警靠 shadow proxy。建议:**跑**——LLM 在新设计里唯一的 alpha 声称仍然值得证伪,只是时间尺度要诚实。

**5. 评估合同:三层门 + T0 预注册决策族 + benchmark window 为 primary。** 建议:**接受**。草案自己的 n=60 LB95<0 版本被审查算出 70-95% 误杀——和它嘲讽的旧门一样;修订版 kill-gate(< −0.10R)在声明 mid-band 下误杀 ≈5%,并预注册 60/40 domination 出口。**诚实日历:n=30 健康检查 ≈ 第 4 个月,n=60 kill 检查 ≈ 第 6-7 个月**——不是草案的 8-12 周。

**6.(新增)风险 cap 重构:R2 单一 6 仓上限 → per-sleeve 预算(R2a 指数 vertical ≤9 且 aggregate max_loss ≤15% equity;R2b 股票 ≤4;R2c 其余 ≤2),白名单 cooldown/same-underlying 改 per-expiry。** 建议:**批准**。这是唯一让声明节奏在算术上成立的途径;有界 max_loss 是提额的正当性来源。**若不批**:全书节奏 ≈ 2.4 平仓/周,n=60 在 ~6 个月+,所有 checkpoint 日期按此重推——方案仍成立,只是更慢,且 sleeve 2 基本开不了仓。

---

## 附:执行顺序总览(修订)

```
Week 1-2:  jennings merge → migration 014 (EC2, lehmann push 前) → lehmann merge
           (+combo-leg patch, 验收改 dry-run, 不开真实 vertical)
           → rvol fix (time-of-day 基线 + prompt 重校, 同一 commit)
           → 双窗口 timer (America/New_York) + decline-cooldown 同批
           → schema bands / qty-0 事件 / 0.50 阈值预注册规则
           → deadman + −2R 熔断 (sleeve 1 开仓前置)
Month 1:   M1-0 阻塞前提: combo-close 路径 → combo 成本模型 → per-underlying
           点差校准 (IWM 去留) → gate 可行性快照回放 + managed-payoff 期望
           → M1-2 per-sleeve caps (Ask #6) → M1-3 评估合同 + T0 预注册
           → M1-4 入场路径 5 组件 → sleeve 1 首笔 fill (全绿后, ≈第 5-6 周)
Month 2:   insider_cluster_stock (含 per-insider 3-5y Form-4 历史抓取)
           → convexity shadow log 稳态运行
Quarter:   n=30 健康检查 (≈第 4 个月) → benchmark domination checkpoint
           → learning 参数族指向新 sleeve → earnings_dte 接入
           → 删除清单执行 → n=60 kill-gate 首查 (≈第 6-7 个月)
```

系统的病是三层叠加:唯一的眼睛瞎了、唯一的策略是文献级输家画像、唯一的评估门数学上打不出"通过"。草案的解在第三层差点重蹈覆辙(新门 70-95% 误杀)、在第一层差点从"全压制"翻成"全报警"(均匀 elapsed 归一化)、在第二层漏掉了 edge 的执行前提(combo 平不了仓)。本 FINAL 版三处都已按审查修正,并把每一个时间承诺重算到系统自己的硬约束之下。