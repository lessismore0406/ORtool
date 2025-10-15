# app.py
import streamlit as st
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

st.set_page_config(page_title="OR-Tools Routing Demos", layout="wide")
st.title("OR-Tools 路由求解器 · 经典案例合集")
st.caption("包含：简单VRP / 载重限制CVRP / 取货-送货PD / 时间窗VRPTW / 多起点多终点")

# ---------- 工具：通用求解输出 ----------
def extract_routes(manager, routing, solution, extra=None):
    routes = []
    for v in range(routing.vehicles()):
        idx = routing.Start(v)
        seq = []
        load_seq = []
        time_seq = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            seq.append(node)
            if extra and extra.get("dim"):
                dim = routing.GetDimensionOrDie(extra["dim"])
                time_seq.append(solution.Value(dim.CumulVar(idx)))
            if extra and extra.get("load_dim"):
                ldim = routing.GetDimensionOrDie(extra["load_dim"])
                load_seq.append(solution.Value(ldim.CumulVar(idx)))
            idx = solution.Value(routing.NextVar(idx))
        seq.append(manager.IndexToNode(idx))
        if extra and extra.get("dim"):
            dim = routing.GetDimensionOrDie(extra["dim"])
            time_seq.append(solution.Value(dim.CumulVar(idx)))
        if extra and extra.get("load_dim"):
            ldim = routing.GetDimensionOrDie(extra["load_dim"])
            load_seq.append(solution.Value(ldim.CumulVar(idx)))
        routes.append({"vehicle": v, "nodes": seq, "time": time_seq, "load": load_seq})
    return routes

def show_routes(routes, show_time=False, show_load=False, title=None):
    if title:
        st.subheader(title)
    for r in routes:
        df = pd.DataFrame({"step": range(len(r["nodes"])), "node": r["nodes"]})
        if show_time and r["time"]:
            df["cumul_time"] = r["time"]
        if show_load and r["load"]:
            df["cumul_load"] = r["load"]
        st.markdown(f"**车辆 {r['vehicle']}**")
        st.dataframe(df, use_container_width=True)

# ---------- 1) 简单 VRP ----------
def demo_simple_vrp(num_vehicles=2, depot=0):
    time_matrix = [
        [0, 4, 8, 8, 7, 3],
        [4, 0, 6, 5, 3, 5],
        [8, 6, 0, 3, 4, 6],
        [8, 5, 3, 0, 2, 7],
        [7, 3, 4, 2, 0, 6],
        [3, 5, 6, 7, 6, 0],
    ]
    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return time_matrix[i][j]

    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(params)
    if not sol: return None
    return extract_routes(manager, routing, sol)

# ---------- 2) 载重限制 CVRP ----------
def demo_cvrp(num_vehicles=2, depot=0, vehicle_capacity=15):
    time_matrix = [
        [0, 4, 8, 8, 7, 3],
        [4, 0, 6, 5, 3, 5],
        [8, 6, 0, 3, 4, 6],
        [8, 5, 3, 0, 2, 7],
        [7, 3, 4, 2, 0, 6],
        [3, 5, 6, 7, 6, 0],
    ]
    demands = [0, 4, 6, 3, 4, 5]  # 与节点数对应，0 是仓库

    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return time_matrix[i][j]
    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    # 载重维度：累积需求不能超过车辆容量
    def demand_cb(index):
        return demands[manager.IndexToNode(index)]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,  # 无额外 slack
        [vehicle_capacity] * num_vehicles,
        True,
        "Load",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(params)
    if not sol: return None
    return extract_routes(manager, routing, sol, extra={"load_dim": "Load"})

# ---------- 3) 取货送货 PD ----------
def demo_pickup_delivery(num_vehicles=2, depot=0):
    time_matrix = [
        [0, 9, 9, 9, 6, 6, 6],
        [9, 0, 4, 4, 6, 6, 6],
        [9, 4, 0, 3, 6, 6, 6],
        [9, 4, 3, 0, 6, 6, 6],
        [6, 6, 6, 6, 0, 2, 2],
        [6, 6, 6, 6, 2, 0, 2],
        [6, 6, 6, 6, 2, 2, 0],
    ]
    # 成对的 (pickup, delivery) 索引
    pairs = [(1, 4), (2, 5), (3, 6)]

    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return time_matrix[i][j]
    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    # 添加取货-送货对 + 同车 + 先取后送约束
    for p, d in pairs:
        p_i, d_i = manager.NodeToIndex(p), manager.NodeToIndex(d)
        routing.AddPickupAndDelivery(p_i, d_i)
        routing.solver().Add(routing.VehicleVar(p_i) == routing.VehicleVar(d_i))
        routing.solver().Add(routing.CumulVar(p_i, "Time") <= routing.CumulVar(d_i, "Time"))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(params)
    if not sol: return None
    return extract_routes(manager, routing, sol)

# ---------- 4) 时间窗 VRPTW ----------
def demo_time_windows(num_vehicles=2, depot=0, max_wait=30, horizon=60):
    time_matrix = [
        [0, 9, 9, 9, 9, 9],
        [9, 0, 6, 6, 6, 6],
        [9, 6, 0, 4, 4, 4],
        [9, 6, 4, 0, 3, 3],
        [9, 6, 4, 3, 0, 2],
        [9, 6, 4, 3, 2, 0],
    ]
    time_windows = [
        (0, 30),  # depot
        (0, 15),
        (3, 18),
        (6, 24),
        (0, 30),
        (0, 30),
    ]

    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return time_matrix[i][j]
    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    routing.AddDimension(cb, max_wait, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node, (start, end) in enumerate(time_windows):
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(start, end)
    for v in range(num_vehicles):
        s, e = routing.Start(v), routing.End(v)
        depot_start, depot_end = time_windows[depot]
        time_dim.CumulVar(s).SetRange(depot_start, depot_end)
        time_dim.CumulVar(e).SetRange(0, horizon)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(params)
    if not sol: return None
    return extract_routes(manager, routing, sol, extra={"dim": "Time"})

# ---------- 5) 多起点多终点 ----------
def demo_multiple_starts_ends():
    time_matrix = [
        [0, 4, 8, 8, 7, 3],
        [4, 0, 6, 5, 3, 5],
        [8, 6, 0, 3, 4, 6],
        [8, 5, 3, 0, 2, 7],
        [7, 3, 4, 2, 0, 6],
        [3, 5, 6, 7, 6, 0],
    ]
    # 2 辆车：车辆0 从 0 出发到 4 结束；车辆1 从 5 出发到 1 结束
    starts = [0, 5]
    ends   = [4, 1]

    manager = pywrapcp.RoutingIndexManager(len(time_matrix), len(starts), starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return time_matrix[i][j]
    cb = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.time_limit.FromSeconds(5)

    sol = routing.SolveWithParameters(params)
    if not sol: return None
    return extract_routes(manager, routing, sol)

# ========== UI ==========
case = st.selectbox(
    "选择案例",
    ["简单VRP", "载重限制CVRP", "取货-送货PD", "时间窗VRPTW", "多起点多终点"],
)

if case == "简单VRP":
    nv = st.sidebar.number_input("车辆数", 1, 5, 2)
    if st.button("求解"):
        routes = demo_simple_vrp(nv)
        st.success("完成！"); show_routes(routes, title="解的明细")

elif case == "载重限制CVRP":
    nv = st.sidebar.number_input("车辆数", 1, 5, 2)
    cap = st.sidebar.number_input("车辆容量", 1, 100, 15)
    if st.button("求解"):
        routes = demo_cvrp(nv, vehicle_capacity=cap)
        st.success("完成！"); show_routes(routes, show_load=True, title="解的明细（含累计载重）")

elif case == "取货-送货PD":
    nv = st.sidebar.number_input("车辆数", 1, 5, 2)
    if st.button("求解"):
        routes = demo_pickup_delivery(nv)
        st.success("完成！"); show_routes(routes, title="解的明细（取后送）")

elif case == "时间窗VRPTW":
    nv = st.sidebar.number_input("车辆数", 1, 5, 2)
    max_wait = st.sidebar.number_input("最大等待(slack)", 0, 600, 30)
    horizon  = st.sidebar.number_input("时间上界", 1, 1000, 60)
    if st.button("求解"):
        routes = demo_time_windows(nv, max_wait=max_wait, horizon=horizon)
        st.success("完成！"); show_routes(routes, show_time=True, title="解的明细（含累计时间）")

else:  # 多起点多终点
    if st.button("求解"):
        routes = demo_multiple_starts_ends()
        st.success("完成！"); show_routes(routes, title="解的明细（多起终点）")