# -*- coding: utf-8 -*-
"""
层级 GCML 最小实验骨架 (Hierarchical GCML — minimal experiment skeleton)
========================================================================

同时使用多个不同抽象层级的认知地图, 并加入规则与以往成功解的记忆

两层 GCML (世界层 + 节点层), 与扁平 GCML 对照。

 学习规则忠于原论文
-------------------------------------
1. 环境: "链式小镇" —— n_worlds 个世界, 每世界 n_nodes_per_world 个节点;
   世界内部随机连通, 相邻世界之间由 1 条双向桥连接。
2. 模型: 与论文完全相同的 CML 学习规则 (Q / V / W / G, delta 规则 + Hebbian
   逆模型, 公式(8)-(14)), 纯 NumPy, 无反向传播, 无 torch 依赖。
3. 层级分工:
   - 高层 GCML: 观测 = 世界 one-hot, 动作 = {向左, 向右}, 采样"世界序列";
   - 低层 GCML: 每世界一个, 观测 = 本世界节点 ∪ 桥对侧入口节点,
     动作 = 世界内边 + 出桥动作, 把 agent 送到子目标(桥入口 / 最终目标);
   - 执行器: 高层采样世界序列 → 低层逐段执行, 每步重新规划(在线重规划)。
4. 对照: 同一张图上训练一个扁平 GCML, 相同采样预算, 比较
   成功率 / 路径长度比(相对 BFS 最短路) / 想象成本(虚拟步数);
   规模扫描用较高噪声 (sigma=0.25) 暴露扁平方法长距离规划时噪声累积的弱点。


import os
from collections import deque

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except Exception:                      # matplotlib 缺失时仍可运行 (只打印指标)
    HAS_MPL = False

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

# ----------------------------------------------------------------------------
# 1. 环境: 层级图 (链式小镇)
# ----------------------------------------------------------------------------

def construct_connected_subgraph(n, rng, extra_edge_p=0.25):
    """n 节点随机连通图: 随机环保底连通 + 随机加边。"""
    adj = np.zeros((n, n), dtype=bool)
    perm = rng.permutation(n)
    for i in range(n):                       # 随机环 -> 保证连通
        a, b = perm[i], perm[(i + 1) % n]
        adj[a, b] = adj[b, a] = True
    for i in range(n):                       # 随机加边
        for j in range(i + 1, n):
            if not adj[i, j] and rng.random() < extra_edge_p:
                adj[i, j] = adj[j, i] = True
    return adj


class HierarchicalGraph:
    """链式小镇: 世界内随机连通, 相邻世界间 1 条双向桥。"""

    def __init__(self, n_worlds, n_nodes_per_world, seed=0, extra_edge_p=0.25):
        rng = np.random.default_rng(seed)
        self.n_worlds = n_worlds
        self.m = n_nodes_per_world
        self.N = n_worlds * n_nodes_per_world
        self.world_of = np.repeat(np.arange(n_worlds), n_nodes_per_world)

        adj = np.zeros((self.N, self.N), dtype=bool)
        for w in range(n_worlds):
            s, e = w * self.m, (w + 1) * self.m
            adj[s:e, s:e] = construct_connected_subgraph(self.m, rng, extra_edge_p)

        self.bridges = {}                    # (w, w+1) -> (w 内出口节点, w+1 内入口节点)
        for w in range(n_worlds - 1):
            u = w * self.m + int(rng.integers(self.m))
            v = (w + 1) * self.m + int(rng.integers(self.m))
            adj[u, v] = adj[v, u] = True
            self.bridges[(w, w + 1)] = (u, v)
        self.adj = adj

        # 无向边表 + 有向动作表 (a = 2*edge_id + dir)
        self.edges = []
        self.edge_of = {}
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if adj[i, j]:
                    self.edge_of[(i, j)] = self.edge_of[(j, i)] = len(self.edges)
                    self.edges.append((i, j))
        self.A = 2 * len(self.edges)         # 扁平动作数 (有向边)
        self.action_from = np.empty(self.A, dtype=int)
        self.action_to = np.empty(self.A, dtype=int)
        for e, (i, j) in enumerate(self.edges):
            self.action_from[2 * e], self.action_to[2 * e] = i, j
            self.action_from[2 * e + 1], self.action_to[2 * e + 1] = j, i
        self.node_actions = [[] for _ in range(self.N)]
        for a in range(self.A):
            self.node_actions[self.action_from[a]].append(a)

    def afford_mask(self, node):
        """扁平层: 节点 node 的真实 affordance (该节点出发的有向边)。"""
        mask = np.zeros(self.A, dtype=bool)
        mask[self.node_actions[node]] = True
        return mask


class LowWorld:
    """低层环境描述: 世界 w 的节点 + 桥对侧入口节点, 动作含出桥动作。"""

    def __init__(self, graph, w):
        self.w = w
        g = graph
        s, e = w * g.m, (w + 1) * g.m
        self.obs_nodes = list(range(s, e))   # 本世界节点
        self.bridge_action_target = {}       # action_id -> 入口节点
        edge_pairs = []                      # 全部有向边 (i -> j), i 必在本世界内
        for i in range(s, e):
            for j in np.where(g.adj[i])[0]:
                if s <= j < e:               # 世界内边 (双向各算一次)
                    edge_pairs.append((i, int(j)))
        # 出桥动作: 向左 (w>0) 与 向右 (w<n_worlds-1)
        if w > 0:
            u, v = g.bridges[(w - 1, w)]     # u 在世界 w-1, v 在世界 w
            self.obs_nodes.append(u)
            edge_pairs.append((v, u))
        if w < g.n_worlds - 1:
            u, v = g.bridges[(w, w + 1)]     # u 在世界 w, v 在世界 w+1
            self.obs_nodes.append(v)
            edge_pairs.append((u, v))
        self.obs_id = {n: i for i, n in enumerate(self.obs_nodes)}
        self.A = len(edge_pairs)
        self.action_of = {}
        self.action_from = np.empty(self.A, dtype=int)
        self.action_to = np.empty(self.A, dtype=int)
        for a, (i, j) in enumerate(edge_pairs):
            self.action_of[(i, j)] = a
            self.action_from[a] = self.obs_id[i]
            self.action_to[a] = self.obs_id[j]
        self.node_actions = [[] for _ in range(len(self.obs_nodes))]
        for a in range(self.A):
            self.node_actions[self.action_from[a]].append(a)

    def afford_mask(self, obs_node):
        """低层: 观测节点 obs_node 的 affordance。"""
        mask = np.zeros(self.A, dtype=bool)
        mask[self.node_actions[obs_node]] = True
        return mask


def high_afford_mask(world, n_worlds):
    """高层: 世界 world 的 {左, 右} affordance。"""
    return np.array([world > 0, world < n_worlds - 1], dtype=bool)


def entry_node(graph, w_from, w_to):
    """从世界 w_from 出发、进入世界 w_to 的桥入口节点 (位于 w_to 内)。"""
    if w_to == w_from + 1:
        return graph.bridges[(w_from, w_to)][1]
    return graph.bridges[(w_to, w_from)][0]


def bfs_shortest(adj, s, t):
    """无权最短路长度 (边数); 不连通返回 None。"""
    dist = {s: 0}
    q = deque([s])
    while q:
        x = q.popleft()
        if x == t:
            return dist[x]
        for y in np.where(adj[x])[0]:
            y = int(y)
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return None


# ----------------------------------------------------------------------------
# 2. GCML 模型 (与论文相同的 Q/V/W/G 学习规则, 纯 NumPy)
# ----------------------------------------------------------------------------

class GCML:
    """生成式认知地图学习器: 学习嵌入 + 目标导向采样。"""

    def __init__(self, n_obs, n_actions, s_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.Q = rng.normal(0, 1.0, (s_dim, n_obs))      # 观测嵌入, 公式(8)
        self.V = rng.normal(0, 0.1, (s_dim, n_actions))  # 动作嵌入(前向), 公式(10)
        self.W = rng.normal(0, 0.1, (n_actions, s_dim))  # 逆模型, 公式(14)
        self.G = rng.normal(0, 0.1, (n_actions, s_dim))  # affordance 门控, 公式(17)
        self.A = n_actions
        self.rng = np.random.default_rng(seed + 1)       # 采样噪声独立于初始化

    def update(self, o_pre, a, o_next, g_true,
               lr_q=0.1, lr_v=0.01, lr_w=0.01, lr_g=0.0005):
        """一批 (o_pre, a, o_next) 上的局部突触更新, 返回批预测误差 (监控用)。"""
        o_pre = np.asarray(o_pre, dtype=int)
        a = np.asarray(a, dtype=int)
        o_next = np.asarray(o_next, dtype=int)
        B = o_pre.shape[0]

        s_pre = self.Q[:, o_pre]
        s_next = self.Q[:, o_next]
        s_pred = s_pre + self.V[:, a]
        pe = s_next - s_pred                     # 状态预测误差

        np.add.at(self.Q, (slice(None), o_next), -lr_q * pe)   # 公式(13)
        np.add.at(self.V, (slice(None), a), lr_v * pe)         # 公式(12)

        ds = s_next - s_pre                      # 局部状态差
        u = self.W @ ds
        onehot = np.zeros((self.A, B))
        onehot[a, np.arange(B)] = 1.0
        self.W += lr_w * (onehot - u) @ ds.T / B               # Hebbian 逆模型
        g_est = self.G @ s_pre
        self.G += lr_g * (g_true - g_est) @ s_pre.T / B        # affordance 学习
        return float(np.mean(pe ** 2))

    def plan(self, start, goal, afford_fn, T=20, n_samples=5, sigma=0.15):
        """目标导向采样 (公式(15)(16)(18)(19)(20))。

        返回 (最优想象轨迹 [观测id列表], 虚拟步成本);
        没有样本到达目标时返回 (None, 成本)。
        """
        s_goal = self.Q[:, goal]
        best, cost = None, 0
        for _ in range(n_samples):
            s_hat = self.Q[:, start].copy()
            cur, path, arrived = start, [start], False
            for t in range(T):
                cost += 1
                u = self.W @ (s_goal - s_hat)                  # 效用
                if t == 0:
                    u[~afford_fn(cur)] = -np.inf               # 真实 affordance 门控
                else:
                    u = u * (self.G @ s_hat)                   # 学习型软门控
                scale = np.abs(u[np.isfinite(u)]).max()        # 噪声尺度只取有限值
                u = u + sigma * scale * self.rng.standard_normal(self.A)
                a = int(np.argmax(u))                          # WTA 动作选择
                s_hat = s_hat + self.V[:, a]                   # 自举 (公式20)
                # 想象中无环境反馈: 用最近邻节点判"走到哪/是否到达"(简化, 见 docstring)
                cur = int(np.argmin(np.sum((self.Q - s_hat[:, None]) ** 2, axis=0)))
                path.append(cur)
                if cur == goal:
                    arrived = True
                    break
            if arrived and (best is None or len(path) < len(best)):
                best = path
        return (best, cost) if best is not None else (None, cost)


# ----------------------------------------------------------------------------
# 3. 训练数据: 随机游走轨迹
# ----------------------------------------------------------------------------

def gen_walk_transitions(env, n_traj, length, rng, start_sampler, action_sampler):
    """通用随机游走生成 (o_pre, a, o_next) 序列; 轨迹可提前终止 (桥外逃)。"""
    trajs = []
    for _ in range(n_traj):
        o_pre, a_, o_next = [], [], []
        cur = start_sampler()
        for _ in range(length):
            actions = action_sampler(cur)
            if not actions:
                break
            a = int(actions[rng.integers(len(actions))])
            nxt = env.action_to[a]
            o_pre.append(cur)
            a_.append(a)
            o_next.append(nxt)
            cur = nxt
        if o_pre:
            trajs.append((np.array(o_pre), np.array(a_), np.array(o_next)))
    return trajs


def train(gcml, trajs, env, epochs, rng, label="", print_every=300):
    """遍历轨迹做批更新; g_true 由环境真实 affordance 构造。"""
    losses = []
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for o_pre, a, o_next in trajs:
            g_true = np.zeros((gcml.A, len(o_pre)))
            for b, x in enumerate(o_pre):
                g_true[env.afford_mask(x), b] = 1.0
            tot += gcml.update(o_pre, a, o_next, g_true)
            cnt += 1
        losses.append(tot / max(cnt, 1))
        if print_every and (ep + 1) % print_every == 0:
            print(f"  [{label}] epoch {ep+1}/{epochs}  mean PE^2 = {losses[-1]:.4f}")
    return losses


# ----------------------------------------------------------------------------
# 4. 执行器: 扁平 vs 层级
# ----------------------------------------------------------------------------

def run_flat(flat, graph, start, goal, T, S, sigma, max_steps):
    """扁平 GCML: 每步用 GCML 想象 T 步 × S 样本, 执行首个动作。"""
    cur, path, cost = start, [start], 0
    for _ in range(max_steps):
        p, c = flat.plan(cur, goal, graph.afford_mask, T, S, sigma)
        cost += c
        if p is None or len(p) < 2:
            return path, False, cost
        cur = p[1]
        path.append(cur)
        if cur == goal:
            return path, True, cost
    return path, False, cost


def run_hierarchical(graph, high, lows, start, goal, T, S, sigma, max_steps):
    """层级 GCML: 高层采样世界序列, 低层逐段执行到桥入口, 最后到目标。"""
    w_start, w_goal = int(graph.world_of[start]), int(graph.world_of[goal])
    cur, path, cost = start, [start], 0

    if w_start == w_goal:
        seq = [w_start]
    else:
        p, c = high.plan(w_start, w_goal,
                         lambda w: high_afford_mask(w, graph.n_worlds), T, S, sigma)
        cost += c
        if p is None:
            return path, False, cost
        seq = p

    for k in range(len(seq) - 1):                        # 逐段: 当前世界 -> 桥入口
        w_from, w_to = seq[k], seq[k + 1]
        sub = entry_node(graph, w_from, w_to)
        lw, model = lows[w_from]
        for _ in range(max_steps):
            if cur == sub:
                break
            p, c = model.plan(lw.obs_id[cur], lw.obs_id[sub],
                              lw.afford_mask, T, S, sigma)
            cost += c
            if p is None or len(p) < 2:
                return path, False, cost
            cur = lw.obs_nodes[p[1]]
            path.append(cur)
        if cur != sub:
            return path, False, cost

    lw, model = lows[w_goal]                             # 末段: 目标世界内到目标
    for _ in range(max_steps):
        if cur == goal:
            return path, True, cost
        p, c = model.plan(lw.obs_id[cur], lw.obs_id[goal],
                          lw.afford_mask, T, S, sigma)
        cost += c
        if p is None or len(p) < 2:
            return path, False, cost
        cur = lw.obs_nodes[p[1]]
        path.append(cur)
    return path, cur == goal, cost


# ----------------------------------------------------------------------------
# 5. 实验协议
# ----------------------------------------------------------------------------

def build_all_models(graph, seed=0, s_flat=64, s_high=16, s_low=32,
                     ep_flat=1500, ep_high=800, ep_low=800):
    """构造 + 训练: 1 个扁平 GCML、1 个高层 GCML、每世界 1 个低层 GCML。"""
    rng = np.random.default_rng(seed)
    pe = max(300, min(ep_flat, ep_high, ep_low) // 2)

    flat = GCML(graph.N, graph.A, s_flat, seed=seed)
    trajs = gen_walk_transitions(
        graph, n_traj=200, length=32, rng=rng,
        start_sampler=lambda: int(rng.integers(graph.N)),
        action_sampler=lambda x: graph.node_actions[x])
    print("训练 扁平 GCML ...")
    train(flat, trajs, graph, epochs=ep_flat, rng=rng, label="flat", print_every=pe)

    high = GCML(graph.n_worlds, 2, s_high, seed=seed + 100)
    ht = []
    for _ in range(150):
        o_pre, a_, o_next, cur = [], [], [], int(rng.integers(graph.n_worlds))
        for _ in range(8):
            avail = [d for d in (-1, 1) if 0 <= cur + d < graph.n_worlds]
            d = avail[int(rng.integers(len(avail)))]
            a = 0 if d == -1 else 1                       # 0=左, 1=右
            o_pre.append(cur); a_.append(a); o_next.append(cur + d)
            cur = cur + d
        ht.append((np.array(o_pre), np.array(a_), np.array(o_next)))
    print("训练 高层 GCML (世界层) ...")
    high_env = _HighEnv(graph.n_worlds)
    train(high, ht, high_env, epochs=ep_high, rng=rng, label="high", print_every=pe)

    lows = {}
    for w in range(graph.n_worlds):
        lw = LowWorld(graph, w)
        lt = gen_walk_transitions(
            lw, n_traj=120, length=16, rng=rng,
            start_sampler=lambda w=w, m=graph.m, lw=lw:
                lw.obs_id[int(rng.integers(w * m, (w + 1) * m))],
            action_sampler=lambda x, lw=lw: lw.node_actions[x])
        gcml = GCML(len(lw.obs_nodes), lw.A, s_low, seed=seed + 200 + w)
        train(gcml, lt, lw, epochs=ep_low, rng=rng, label=f"low[{w}]", print_every=pe)
        lows[w] = (lw, gcml)
    return flat, high, lows


class _HighEnv:
    """高层环境的轻量包装 (train/规划共用 affordance 接口)。"""
    def __init__(self, n_worlds):
        self.n_worlds = n_worlds
        self.action_to = None                      # 高层轨迹由 hand-made walk 生成

    def afford_mask(self, world):
        return high_afford_mask(world, self.n_worlds)


def evaluate(graph, flat, high, lows, n_pairs, seed=0, T=20, S=5, sigma=0.15):
    """随机 (start, goal) 跨世界对上的扁平 vs 层级对照。"""
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        s = int(rng.integers(graph.N))
        t = int(rng.integers(graph.N))
        if graph.world_of[s] != graph.world_of[t]:
            d = bfs_shortest(graph.adj, s, t)
            if d is not None:
                pairs.append((s, t, d))

    res = {"flat": {"succ": [], "ratio": [], "cost": []},
           "hier": {"succ": [], "ratio": [], "cost": []}}
    for s, t, d in pairs:
        max_steps = 5 * d
        pf, ok_f, cf = run_flat(flat, graph, s, t, T, S, sigma, max_steps)
        ph, ok_h, ch = run_hierarchical(graph, high, lows, s, t, T, S, sigma, max_steps)
        res["flat"]["succ"].append(ok_f)
        res["flat"]["ratio"].append(len(pf) - 1 - d if ok_f else np.nan)
        res["flat"]["cost"].append(cf)
        res["hier"]["succ"].append(ok_h)
        res["hier"]["ratio"].append(len(ph) - 1 - d if ok_h else np.nan)
        res["hier"]["cost"].append(ch)
    return res


def summarize(res, title=""):
    print(f"\n===== {title} =====")
    for m in ("flat", "hier"):
        succ = np.mean(res[m]["succ"]) * 100
        ratio = [x for x in res[m]["ratio"] if not np.isnan(x)]
        mean_over = np.nanmean(ratio) if ratio else float("nan")
        mean_cost = np.mean(res[m]["cost"])
        print(f"  {m:5s} | 成功率 {succ:5.1f}% | 到达时路径超 BFS 最优 {mean_over:+6.2f} 步"
              f" | 平均想象步数 {mean_cost:8.1f}")
    return {m: (np.mean(res[m]["succ"]) * 100, np.mean(res[m]["cost"])) for m in res}


# ----------------------------------------------------------------------------
# 6. 可视化
# ----------------------------------------------------------------------------

def pca2(X):
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def plot_plans(graph, flat, high, lows, s, t, pf, ph, path_png):
    if not HAS_MPL:
        print("[无 matplotlib, 跳过图 1]")
        return
    coords = pca2(flat.Q.T)                      # 用扁平认知地图的 PCA 作布局
    colors = plt.cm.tab10(np.arange(graph.n_worlds) / max(graph.n_worlds, 1))
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, j in graph.edges:                     # 世界内边
        c = colors[graph.world_of[i]]
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=c, lw=0.8, alpha=0.35)
    for (u, v) in graph.bridges.values():        # 桥 (虚线)
        ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]],
                color="black", lw=2, ls="--", alpha=0.8)
    for w in range(graph.n_worlds):
        idx = np.where(graph.world_of == w)[0]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=60, color=colors[w],
                   edgecolors="white", linewidths=1, label=f"world {w}")
    ax.plot(coords[pf, 0], coords[pf, 1], color="red", lw=2.5, marker="o",
            markersize=3, label="flat plan")
    ax.plot(coords[ph, 0], coords[ph, 1], color="blue", lw=2.5, marker="o",
            markersize=3, label="hierarchical plan")
    ax.scatter(*coords[s], s=220, marker="*", color="black", zorder=5, label="start")
    ax.scatter(*coords[t], s=140, marker="o", color="lime", edgecolors="black",
               zorder=5, label="goal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="best")
    ax.set_title("两级 GCML vs 扁平 GCML (布局 = 扁平认知地图 PCA)")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1 已保存] {path_png}")


def plot_scaling(n_worlds_list, succ_flat, succ_hier, path_png):
    if not HAS_MPL:
        print("[无 matplotlib, 跳过图 2]")
        return
    x = np.arange(len(n_worlds_list))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, succ_flat, w, label="flat GCML", color="crimson")
    ax.bar(x + w / 2, succ_hier, w, label="hierarchical GCML", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} 世界\n({n*6} 节点)" for n in n_worlds_list], fontsize=9)
    ax.set_ylabel("成功率 (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("层级结构随规模增长的收益")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2 已保存] {path_png}")


# ----------------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("层级 GCML 最小实验骨架 | Lin et al. Nat. Mach. Intell. 2026 (Discussion 开放问题)")
    print("=" * 70)

    # ---- 基准配置: 4 世界 × 6 节点 ----
    T, S, sigma = 20, 5, 0.15
    g = HierarchicalGraph(n_worlds=4, n_nodes_per_world=6, seed=0)
    print(f"\n环境: {g.n_worlds} 世界 × {g.m} 节点 = {g.N} 节点, "
          f"扁平动作数 {g.A}, 桥: {g.bridges}")
    flat, high, lows = build_all_models(g, seed=0)
    res = evaluate(g, flat, high, lows, n_pairs=10, seed=1, T=T, S=S, sigma=sigma)
    summarize(res, title=f"基准配置 ({g.N} 节点, sigma={sigma}, 每步 {T} 想象步 × {S} 样本)")

    rng = np.random.default_rng(7)
    s = int(rng.integers(g.N))
    t = int(rng.integers(g.N))
    while g.world_of[s] == g.world_of[t]:
        t = int(rng.integers(g.N))
    d = bfs_shortest(g.adj, s, t)
    pf, _, _ = run_flat(flat, g, s, t, T, S, sigma, 5 * d)
    ph, _, _ = run_hierarchical(g, high, lows, s, t, T, S, sigma, 5 * d)
    plot_plans(g, flat, high, lows, s, t, pf, ph,
               os.path.join(FIG_DIR, "hier_gcml_plans.png"))
    print(f"[示例] start={s}(世界{g.world_of[s]}) -> goal={t}(世界{g.world_of[t]}), "
          f"BFS 最短 {d} 步 | 扁平路径 {len(pf)-1} 步 | 层级路径 {len(ph)-1} 步")

    # ---- 规模扫描: 层级结构的收益 (更高噪声暴露扁平方法的长距离弱点) ----
    n_worlds_list = [3, 5, 7, 9]
    sigma_sweep = 0.25
    succ_flat, succ_hier = [], []
    for nw in n_worlds_list:
        gs = HierarchicalGraph(n_worlds=nw, n_nodes_per_world=6, seed=nw)
        print(f"\n--- 规模扫描: {nw} 世界 ({gs.N} 节点, sigma={sigma_sweep}) ---")
        fs, hs, ls = build_all_models(gs, seed=nw * 10,
                                      ep_flat=1200, ep_high=600, ep_low=600)
        r = evaluate(gs, fs, hs, ls, n_pairs=8, seed=nw,
                     T=T, S=S, sigma=sigma_sweep)
        out = summarize(r, title=f"{gs.N} 节点")
        succ_flat.append(out["flat"][0])
        succ_hier.append(out["hier"][0])
    plot_scaling(n_worlds_list, succ_flat, succ_hier,
                 os.path.join(FIG_DIR, "hier_gcml_scaling.png"))
    print(f"\n[规模扫描使用 sigma={sigma_sweep}, 以暴露扁平 GCML 在长距离规划上的弱点]")

    print("\n完成。TODO 扩展点见文件头部 docstring。")


if __name__ == "__main__":
    main()
