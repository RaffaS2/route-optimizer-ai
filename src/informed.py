"""
informed.py
===========
Pesquisa informada: Greedy Best-First e A*, com duas heurísticas.

O problema é resolvido em duas fases:
  1. Atribuição: que lotes de recolha (produtor, dia) faz cada veículo.
     Neste problema é feita pelo CSP (csp.py); aqui há também uma regra gulosa
     simples (greedy_assignment) para mostrar porque é que não chega.
  2. Encaminhamento: para cada veículo, a ordem das paragens, com
     Greedy Best-First ou A* (route_search).

Espaço de estados de um veículo:
  * estado inicial : veículo no armazém, nenhuma tarefa feita;
  * ações          : fazer uma das tarefas que faltam (recolha ou entrega);
  * transição      : model.step (simula viagem, esperas, armazém, ...);
  * estado objetivo: todas as tarefas feitas e veículo de volta ao armazém;
  * custo g(n)     : km percorridos até ao estado n.
Estados inviáveis (janela falhada, capacidade, ...) não são gerados.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass

from model import Problem, Vehicle, Plan, step, finish


# ---------------------------------------------------------------------------
# Heurísticas: estimativa (por baixo) dos km que ainda faltam
# ---------------------------------------------------------------------------
def h_farthest(pb: Problem, veh: Vehicle, loc: int, remaining) -> float:
    """h1 - 'paragem mais distante': o veículo ainda tem de ir a cada paragem que
    falta e depois voltar ao armazém. Logo, faltam pelo menos
        max_s  d(atual, s) + d(s, armazém)  km.
    Admissível (desigualdade triangular)."""
    W = veh.depot
    if not remaining:
        return pb.dist[loc, W]
    return max(pb.dist[loc, t.loc] + pb.dist[t.loc, W] for t in remaining)


def h_mst(pb: Problem, veh: Vehicle, loc: int, remaining) -> float:
    """h2 - 'árvore geradora mínima' (MST): o resto da rota é um caminho que liga
    a posição atual, todas as paragens que faltam e o armazém. Um caminho é uma
    árvore, por isso custa pelo menos o mesmo que a MST desses pontos.
    Admissível. Calculada com o algoritmo de Prim."""
    nodes = list({loc, veh.depot, *(t.loc for t in remaining)})
    best = {n: pb.dist[nodes[0], n] for n in nodes[1:]}   # ligação mais barata à árvore
    total = 0.0
    while best:
        n = min(best, key=best.get)
        total += best.pop(n)
        for m in best:
            best[m] = min(best[m], pb.dist[n, m])
    return total


HEURISTICS = {"h1_farthest": h_farthest, "h2_mst": h_mst}


# ---------------------------------------------------------------------------
# Pesquisa para um veículo
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    route: list | None        # chaves das tarefas por ordem (None se não encontrou)
    cost: float               # km
    expanded: int             # nós expandidos
    generated: int            # nós gerados
    seconds: float


def still_reachable(pb: Problem, s, remaining) -> bool:
    """Poda: o tempo só avança, por isso se uma paragem que falta já não pode ser
    alcançada antes de a sua janela fechar (nem indo diretamente), este estado é
    um beco sem saída."""
    return all(s.t + pb.time[s.loc, t.loc] <= t.end + 1e-6 for t in remaining)


def route_search(pb: Problem, veh: Vehicle, requests, algorithm="astar",
                 heuristic="h2_mst", max_expanded=200_000, dwell=True) -> SearchResult:
    """Greedy Best-First (f = h) ou A* (f = g + h) para a rota de UM veículo.
    dwell=False ignora a regra das 24h no armazém (usado pelo CSP para podar)."""
    t0 = time.perf_counter()
    h = HEURISTICS[heuristic]
    tasks = pb.tasks_for(requests)
    counter = itertools.count()                     # desempate estável na fila

    def f(state, remaining):
        hv = h(pb, veh, state.loc, remaining)
        return state.dist + hv if algorithm == "astar" else hv

    start = pb.initial_state(veh)
    frontier = [(f(start, list(tasks.values())), 0, next(counter), start, False)]
    best_g = {start.signature(): 0.0}               # para não repetir estados
    expanded = generated = 0

    while frontier:
        _, _, _, s, is_goal = heapq.heappop(frontier)
        if is_goal:                                  # objetivo: rota completa e de volta ao armazém
            return SearchResult(list(s.route), s.dist, expanded, generated,
                                time.perf_counter() - t0)
        if s.dist > best_g.get(s.signature(), math.inf) + 1e-9:
            continue                                 # já havia um caminho melhor para este estado
        expanded += 1
        if expanded > max_expanded:
            break

        done = set(s.route)
        remaining = [t for k, t in tasks.items() if k not in done]
        if not remaining:                            # falta só voltar ao armazém
            end = finish(pb, veh, s)
            heapq.heappush(frontier, (end.dist if algorithm == "astar" else 0.0,
                                      0, next(counter), end, True))
            continue

        for task in remaining:                       # gerar sucessores
            child, _ = step(pb, veh, s, task, dwell=dwell)
            if child is None:                        # viola restrições -> podar
                continue
            rest = [t for t in remaining if t is not task]
            if not still_reachable(pb, child, rest):  # alguma paragem já é impossível -> podar
                continue
            sig = child.signature()
            if child.dist >= best_g.get(sig, math.inf) - 1e-9:
                continue
            best_g[sig] = child.dist
            generated += 1
            # desempate: preferir nós mais profundos (mais tarefas feitas)
            heapq.heappush(frontier, (f(child, rest), -len(child.route), next(counter), child, False))

    return SearchResult(None, math.inf, expanded, generated, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Atribuição gulosa (sem retrocesso)
# ---------------------------------------------------------------------------
def greedy_assignment(pb: Problem, heuristic="h2_mst", max_expanded=20_000):
    """Percorre os lotes por ordem cronológica e dá cada um ao veículo onde fica
    mais barato (menos km a mais), desde que a rota continue viável.
    Nunca volta atrás: se um lote não couber em nenhum veículo, falha.
    Devolve ({vehicle_id: [lotes]}, lote_que_falhou_ou_None)."""
    assign = {v.vehicle_id: [] for v in pb.vehicles}
    cost = {v.vehicle_id: 0.0 for v in pb.vehicles}
    for g in pb.groups:
        options = []
        for v in pb.vehicles:
            vid = v.vehicle_id
            reqs = pb.requests_of(assign[vid] + [g])
            res = route_search(pb, v, reqs, "greedy", heuristic, max_expanded)
            if res.route is not None:
                options.append((round(res.cost - cost[vid], 6), len(assign[vid]), vid, res.cost))
        if not options:
            return assign, g
        _, _, vid, new_cost = min(options)
        assign[vid].append(g)
        cost[vid] = new_cost
    return assign, None


# ---------------------------------------------------------------------------
# Rotas de todos os veículos para uma atribuição
# ---------------------------------------------------------------------------
def solve(pb: Problem, assignment: dict, algorithm="astar", heuristic="h2_mst",
          max_expanded=200_000):
    """assignment = {vehicle_id: [lotes]}. Corre a pesquisa para cada veículo.
    Devolve (solução {vehicle_id: Plan}, lista de estatísticas)."""
    sol, stats = {}, []
    for v in pb.vehicles:
        groups = list(assignment[v.vehicle_id])
        reqs = pb.requests_of(groups)
        res = route_search(pb, v, reqs, algorithm, heuristic, max_expanded)
        sol[v.vehicle_id] = Plan(groups, res.route if res.route is not None else [])
        stats.append({"dataset": pb.name, "algoritmo": algorithm, "heurística": heuristic,
                      "veículo": v.vehicle_id, "paragens": len(pb.tasks_for(reqs)),
                      "km": round(res.cost, 2), "nós expandidos": res.expanded,
                      "nós gerados": res.generated, "tempo (s)": round(res.seconds, 4),
                      "encontrou": res.route is not None})
    return sol, stats
