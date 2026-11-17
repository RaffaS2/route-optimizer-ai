"""
model.py
========
Modelo do problema de recolha e entrega de produtos locais
(Produtor -> Armazém -> Local de entrega).

Este módulo contém:
  * as entidades: Request, Vehicle, Task, State e Plan;
  * a classe Problem, que carrega uma instância a partir dos CSVs;
  * o simulador de rotas (step / finish), que é também o verificador de
    viabilidade (check_route);
  * a função de custo (solution_cost) e o verificador de soluções (check_solution).

Unidades: distâncias em km; tempos em minutos contados a partir das 00:00 do
primeiro dia do horizonte (rules.csv -> start_date).

Decisões de modelação (explicadas no notebook principal):
  1. Um pedido (Request) = uma linha de encomenda (OrderItem): levar um produto
     do produtor até ao local de entrega, passando pelo armazém.
  2. Lote de recolha = todos os pedidos de um produtor num dia. Cada lote é
     recolhido por UM veículo, numa só visita (janela = interseção das janelas).
     A decisão de "que veículo faz o quê" é a atribuição dos lotes aos veículos.
  3. Cada pedido é transportado sempre pelo mesmo veículo: recolhe, leva ao
     armazém e entrega. Itens da mesma encomenda podem vir em veículos diferentes
     (a encomenda chega em duas entregas).
  4. Paragens de um veículo:
       'P' recolha = visitar o produtor e levantar o lote;
       'D' entrega = entregar, de uma vez, os itens de uma encomenda que esse
                     veículo transporta.
  5. O veículo passa no armazém automaticamente quando precisa:
       - antes de uma entrega cujos itens ainda não estão carregados
         (descarrega recolhas: 15 min; carrega: 15 min);
       - no fim de cada dia e no fim da rota.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import pandas as pd

DAY = 24 * 60          # minutos num dia
EPS = 1e-6             # tolerância nas comparações


# ---------------------------------------------------------------------------
# Entidades
# ---------------------------------------------------------------------------
@dataclass
class Request:
    """Pedido de transporte = uma linha de encomenda (OrderItem)."""
    request_id: int            # = order_item_id
    order_id: int
    product: str
    weight: float              # kg
    pickup_loc: int            # location_id do produtor
    pickup_start: float        # janela de disponibilidade no produtor (min)
    pickup_end: float
    delivery_loc: int          # location_id do local de entrega escolhido pelo cliente
    delivery_start: float      # janela de entrega (min)
    delivery_end: float


@dataclass
class Vehicle:
    vehicle_id: int
    plate: str
    capacity: float            # kg
    depot: int                 # location_id do armazém (início e fim das rotas)
    refrigerated: bool = False


@dataclass
class Task:
    """Uma paragem de um veículo.
    kind 'P' (recolha):  key = ('P', loc_produtor, dia)
    kind 'D' (entrega):  key = ('D', order_id)
    [start, end] = janela em que o SERVIÇO pode começar."""
    key: tuple
    kind: str
    loc: int
    start: float
    end: float
    weight: float
    requests: tuple            # pedidos recolhidos / entregues nesta paragem


@dataclass
class State:
    """Situação de um veículo depois de executar uma rota parcial.
    É o 'estado' da pesquisa informada."""
    route: tuple = ()                  # tarefas feitas, por ordem
    loc: int = 0                       # localização atual
    t: float = 0.0                     # relógio (min)
    load: float = 0.0                  # kg de recolhas a bordo (a caminho do armazém)
    dist: float = 0.0                  # km percorridos (= custo g)
    onboard: frozenset = frozenset()   # pedidos recolhidos ainda no veículo
    at_wh: dict = field(default_factory=dict)   # pedido -> hora de chegada ao armazém
    batch: tuple | None = None         # carregamento em curso: (hora_carga, dia, kg)

    def copy(self) -> "State":
        return State(self.route, self.loc, self.t, self.load, self.dist,
                     self.onboard, dict(self.at_wh), self.batch)

    def signature(self):
        """Tudo o que influencia o futuro (menos o custo). Estados com a mesma
        assinatura são equivalentes -> a pesquisa só guarda o mais barato."""
        return (frozenset(self.route), self.loc, round(self.t, 4), round(self.load, 4),
                self.onboard, frozenset(self.at_wh.items()), self.batch)


@dataclass
class Plan:
    """Plano de um veículo: que lotes recolhe e por que ordem faz as paragens."""
    groups: list
    route: list


@dataclass
class RouteResult:
    feasible: bool
    distance: float
    reason: str = ""
    end_time: float = 0.0
    log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Problema (instância carregada dos CSVs)
# ---------------------------------------------------------------------------
class Problem:
    TABLES = ["localities", "producers", "products", "collection_centers", "vehicles",
              "customers", "delivery_locations", "orders", "order_items",
              "locations", "distance_matrix", "rules"]

    def __init__(self, data_dir: str):
        self.name = os.path.basename(os.path.normpath(data_dir))
        self.df = {t: pd.read_csv(os.path.join(data_dir, t + ".csv")) for t in self.TABLES}

        # --- regras -----------------------------------------------------------
        rules = dict(zip(self.df["rules"]["rule"], self.df["rules"]["value"]))
        self.rules = rules
        self.start = pd.Timestamp(rules["start_date"])
        self.service = float(rules["stop_service_min"])            # 10 min por paragem
        self.handling = float(rules["warehouse_handling_min"])     # 15 min descarga / carga
        self.max_dwell = float(rules["max_warehouse_dwell_h"]) * 60

        # --- localizações e matriz de distâncias/tempos -------------------------
        loc = self.df["locations"]
        self.loc_name = dict(zip(loc.location_id, loc.name))
        self.loc_type = dict(zip(loc.location_id, loc.location_type))
        self.coords = {r.location_id: (r.latitude, r.longitude) for r in loc.itertuples()}
        ref = {(r.location_type, r.ref_id): r.location_id for r in loc.itertuples()}

        self.dist, self.time = {}, {}
        for r in self.df["distance_matrix"].itertuples():
            self.dist[r.origin_location_id, r.destination_location_id] = r.distance_km
            self.time[r.origin_location_id, r.destination_location_id] = r.duration_min
        for l in loc.location_id:
            self.dist[l, l] = self.time[l, l] = 0.0

        # --- veículos ----------------------------------------------------------
        self.vehicles = [Vehicle(int(v.vehicle_id), v.plate, float(v.capacity_kg),
                                 ref["collection_center", v.center_id], bool(v.refrigerated))
                         for v in self.df["vehicles"].itertuples()]
        self.vehicle = {v.vehicle_id: v for v in self.vehicles}

        # --- pedidos (um por linha de encomenda) -------------------------------
        prod = self.df["products"].set_index("product_id")
        order_loc = {o.order_id: ref["delivery_location", o.delivery_location_id]
                     for o in self.df["orders"].itertuples()}
        self.requests: dict[int, Request] = {}
        for it in self.df["order_items"].itertuples():
            p = prod.loc[it.product_id]
            self.requests[it.order_item_id] = Request(
                it.order_item_id, it.order_id, p["name"], float(it.weight_kg),
                ref["producer", p["producer_id"]],
                self.minutes(it.availability_start), self.minutes(it.availability_end),
                order_loc[it.order_id],
                self.minutes(it.delivery_start), self.minutes(it.delivery_end))
        self.orders = {}                                   # order_id -> [request_ids]
        for r in self.requests.values():
            self.orders.setdefault(r.order_id, []).append(r.request_id)

        # --- lotes de recolha: (produtor, dia) -> pedidos ----------------------
        self.groups = {}
        for r in sorted(self.requests.values(), key=lambda r: r.request_id):
            self.groups.setdefault((r.pickup_loc, int(r.pickup_start // DAY)), []).append(r.request_id)
        self.groups = dict(sorted(self.groups.items(), key=lambda kv: (kv[0][1], kv[0][0])))

        self._tasks_cache = {}

    # --- utilitários --------------------------------------------------------
    def minutes(self, ts) -> float:
        """Data/hora -> minutos desde o início do horizonte."""
        return (pd.Timestamp(ts) - self.start).total_seconds() / 60

    def clock(self, t: float) -> str:
        """Minutos -> 'dd/mm HH:MM'."""
        if t is None or math.isinf(t):
            return "-"
        return (self.start + pd.Timedelta(minutes=round(t))).strftime("%d/%m %H:%M")

    def label(self, key) -> str:
        """Nome legível de uma tarefa."""
        if key[0] == "P":
            day = (self.start + pd.Timedelta(days=key[2])).strftime("%d/%m")
            return f"Recolha {self.loc_name[key[1]]} ({day})"
        loc = self.requests[self.orders[key[1]][0]].delivery_loc
        return f"Entrega enc.{key[1]} @ {self.loc_name[loc]}"

    def group_label(self, g) -> str:
        day = (self.start + pd.Timedelta(days=g[1])).strftime("%d/%m")
        return f"{self.loc_name[g[0]]} ({day})"

    def weight(self, request_ids) -> float:
        return sum(self.requests[r].weight for r in request_ids)

    def requests_of(self, groups) -> list:
        """Pedidos de um conjunto de lotes."""
        return [r for g in groups for r in self.groups[g]]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "produtores": [len(self.df["producers"])],
            "locais de entrega": [len(self.df["delivery_locations"])],
            "veículos": [len(self.vehicles)],
            "encomendas": [len(self.orders)],
            "pedidos (itens)": [len(self.requests)],
            "lotes de recolha": [len(self.groups)],
            "peso total (kg)": [round(self.weight(self.requests), 1)],
            "capacidade total (kg)": [sum(v.capacity for v in self.vehicles)],
        }, index=[self.name])

    # --- paragens de um veículo ---------------------------------------------
    def tasks_for(self, request_ids) -> dict:
        """Paragens (recolhas + entregas) para transportar os pedidos dados.
        Recolha: uma por lote (produtor, dia); janela = interseção das janelas dos itens.
        Entrega: uma por encomenda (os itens partilham local e janela)."""
        key = frozenset(request_ids)
        if key in self._tasks_cache:
            return self._tasks_cache[key]

        tasks, pick, drop = {}, {}, {}
        for rid in sorted(key):
            r = self.requests[rid]
            pick.setdefault((r.pickup_loc, int(r.pickup_start // DAY)), []).append(r)
            drop.setdefault(r.order_id, []).append(r)

        for (p, day), reqs in sorted(pick.items()):
            start, end = max(r.pickup_start for r in reqs), min(r.pickup_end for r in reqs)
            if start > end:
                raise ValueError(f"lote {self.loc_name[p]} dia {day}: janelas sem interseção")
            tasks["P", p, day] = Task(("P", p, day), "P", p, start, end,
                                      sum(r.weight for r in reqs), tuple(r.request_id for r in reqs))

        for o, reqs in sorted(drop.items()):
            tasks["D", o] = Task(
                ("D", o), "D", reqs[0].delivery_loc,
                max(r.delivery_start for r in reqs), min(r.delivery_end for r in reqs),
                sum(r.weight for r in reqs), tuple(r.request_id for r in reqs))

        self._tasks_cache[key] = tasks
        return tasks

    def initial_state(self, veh: Vehicle) -> State:
        return State(loc=veh.depot)


# ---------------------------------------------------------------------------
# Simulação de uma rota (= verificador de viabilidade)
# ---------------------------------------------------------------------------
def _travel(pb: Problem, s: State, to: int):
    s.dist += pb.dist[s.loc, to]
    s.t += pb.time[s.loc, to]
    s.loc = to


def _go_to_warehouse(pb: Problem, veh: Vehicle, s: State, log=None):
    """Vai ao armazém e descarrega o que recolheu (15 min, se trouxer carga)."""
    if s.loc != veh.depot:
        _travel(pb, s, veh.depot)
        if log is not None:
            log.append(("W", "Armazém (chegada)", s.t, s.load))
    if s.onboard:
        for r in s.onboard:
            s.at_wh[r] = s.t                         # hora de chegada ao armazém
        s.load -= pb.weight(s.onboard)
        s.onboard = frozenset()
        s.t += pb.handling
        if log is not None:
            log.append(("W", "Armazém: descarga", s.t - pb.handling, s.load))
    s.batch = None


def step(pb: Problem, veh: Vehicle, s: State, task: Task, log=None, dwell=True):
    """Aplica uma tarefa ao estado s.

    Devolve (novo_estado, "") se for viável, ou (None, motivo) se violar uma
    restrição: janela temporal, capacidade, fluxo pelo armazém ou tempo máximo
    no armazém (24h). Com dwell=False a regra das 24h não é verificada
    (versão relaxada usada pelo CSP para podar)."""
    s = s.copy()
    max_dwell = pb.max_dwell if dwell else math.inf
    task_day = int(task.start // DAY)

    # Fim do dia: antes de uma tarefa de um dia seguinte, volta ao armazém.
    if task_day > int(s.t // DAY) and s.loc != veh.depot:
        _go_to_warehouse(pb, veh, s, log)

    if task.kind == "P":                                       # ---------- RECOLHA
        _travel(pb, s, task.loc)
        start = max(s.t, task.start)                           # pode esperar pela abertura
        if start > task.end + EPS:
            return None, f"{pb.label(task.key)}: chegaria às {pb.clock(start)}, janela fecha às {pb.clock(task.end)}"
        s.t = start + pb.service
        s.load += task.weight
        if s.load > veh.capacity + EPS:
            return None, f"capacidade excedida ({s.load:.1f} > {veh.capacity:.0f} kg)"
        s.onboard = s.onboard | set(task.requests)
        s.batch = None
        if log is not None:
            log.append((task.key, pb.label(task.key), start, s.load))

    else:                                                      # ---------- ENTREGA
        reqs = task.requests
        if any(r not in s.at_wh and r not in s.onboard for r in reqs):
            return None, f"{pb.label(task.key)}: itens ainda não recolhidos"

        # (a) Se o veículo saiu do armazém só com entregas deste dia, os itens podem
        #     ter sido carregados nessa mesma passagem (não precisa de voltar).
        joined = False
        if s.batch is not None and s.batch[1] == task_day and not s.onboard:
            load_start, _, batch_kg = s.batch
            ready = max(s.at_wh[r] for r in reqs) + pb.handling       # já descarregados?
            deadline = min(s.at_wh[r] for r in reqs) + max_dwell      # no máx. 24h no armazém
            if ready <= load_start + EPS and load_start <= deadline + EPS \
                    and batch_kg + task.weight <= veh.capacity + EPS:
                s.batch = (load_start, task_day, batch_kg + task.weight)
                joined = True

        # (b) Senão vai ao armazém: descarrega o que trouxer e carrega estes itens.
        if not joined:
            _go_to_warehouse(pb, veh, s, log)
            ready = max(s.at_wh[r] for r in reqs) + pb.handling
            deadline = min(s.at_wh[r] for r in reqs) + max_dwell
            jit = task.start - pb.time[veh.depot, task.loc] - pb.handling   # carregar "à justa"
            load_start = max(s.t, ready, min(jit, deadline))
            if load_start > deadline + EPS:
                return None, f"{pb.label(task.key)}: itens ficariam mais de {pb.max_dwell/60:.0f}h no armazém"
            if task.weight > veh.capacity + EPS:
                return None, f"{pb.label(task.key)}: não cabe no veículo"
            s.t = load_start + pb.handling
            s.batch = (load_start, task_day, task.weight)
            if log is not None:
                log.append(("W", "Armazém: carga", load_start, None))

        for r in reqs:
            del s.at_wh[r]
        _travel(pb, s, task.loc)
        start = max(s.t, task.start)
        if start > task.end + EPS:
            return None, f"{pb.label(task.key)}: chegaria às {pb.clock(start)}, janela fecha às {pb.clock(task.end)}"
        s.t = start + pb.service
        if log is not None:
            log.append((task.key, pb.label(task.key), start, None))

    s.route = s.route + (task.key,)
    return s, ""


def finish(pb: Problem, veh: Vehicle, s: State, log=None) -> State:
    """Fim da rota: regressar ao armazém."""
    s = s.copy()
    _go_to_warehouse(pb, veh, s, log)
    return s


def check_route(pb: Problem, veh: Vehicle, requests, route) -> RouteResult:
    """Verificador de viabilidade de UMA rota.

    requests : pedidos que o veículo transporta.
    route    : chaves das paragens, por ordem (tem de ter todas as paragens)."""
    tasks = pb.tasks_for(requests)
    log = []
    if len(set(route)) != len(route):
        return RouteResult(False, math.inf, "paragem repetida")
    if set(route) != set(tasks):
        return RouteResult(False, math.inf, "rota incompleta")
    s = pb.initial_state(veh)
    for key in route:
        if key not in tasks:
            return RouteResult(False, math.inf, f"paragem desconhecida {key}")
        s, why = step(pb, veh, s, tasks[key], log)
        if s is None:
            return RouteResult(False, math.inf, why, log=log)
    s = finish(pb, veh, s, log)
    return RouteResult(True, s.dist, "", s.t, log)


# ---------------------------------------------------------------------------
# Soluções completas: {vehicle_id: Plan}
# ---------------------------------------------------------------------------
def plan_result(pb: Problem, vid: int, plan: Plan) -> RouteResult:
    return check_route(pb, pb.vehicle[vid], pb.requests_of(plan.groups), plan.route)


def solution_cost(pb: Problem, sol: dict) -> float:
    """Função de custo = total de km de todos os veículos (inf se alguma rota for inviável)."""
    total = 0.0
    for vid, plan in sol.items():
        r = plan_result(pb, vid, plan)
        if not r.feasible:
            return math.inf
        total += r.distance
    return total


def check_solution(pb: Problem, sol: dict) -> dict:
    """Verifica uma solução completa: todos os lotes recolhidos exatamente uma vez,
    todos os pedidos entregues e todas as rotas viáveis."""
    problems = []
    if sorted(g for p in sol.values() for g in p.groups) != sorted(pb.groups):
        problems.append("há lotes em falta ou repetidos")
    km = 0.0
    for vid, plan in sol.items():
        r = plan_result(pb, vid, plan)
        if not r.feasible:
            problems.append(f"veículo {vid}: {r.reason}")
        km += r.distance
    owner = {r: vid for vid, p in sol.items() for r in pb.requests_of(p.groups)}
    split = sum(1 for rs in pb.orders.values() if len({owner.get(r) for r in rs}) > 1)
    return {"viável": not problems, "km": round(km, 2) if not problems else math.inf,
            "encomendas divididas": split, "problemas": problems}


def route_table(pb: Problem, vid: int, plan: Plan) -> pd.DataFrame:
    """Horário de uma rota, para mostrar no notebook."""
    r = plan_result(pb, vid, plan)
    tasks = pb.tasks_for(pb.requests_of(plan.groups))
    rows, load = [], 0.0
    for n, (key, name, t, ld) in enumerate(r.log):
        if name == "Armazém: carga":
            # carrega os itens das entregas seguintes (até à próxima paragem que não é entrega)
            for k2, *_ in r.log[n + 1:]:
                if k2 == "W" or k2[0] != "D":
                    break
                load += tasks[k2].weight
        elif key != "W" and key[0] == "D":
            load -= tasks[key].weight
        else:
            load = ld
        rows.append({"paragem": name, "hora": pb.clock(t), "carga (kg)": round(abs(load), 1)})
    return pd.DataFrame(rows)
