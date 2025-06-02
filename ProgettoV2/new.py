# === AI Simulation: Kira vs Detectives (Refactored) ===
# Inspired by Death Note - Clean Rebuild
# Focused on chapters 1–18 of Russell & Norvig AI textbook

import streamlit as st
import networkx as nx
import random
import plotly.graph_objects as go
from pyswip import Prolog
import copy
import heapq
from collections import defaultdict, Counter
import pandas as pd
import plotly.express as px

# === Utilities ===
def rnd(a=0.0, b=1.0):
    return random.uniform(a, b)

# === Social Network Class ===
class SocialNetwork:
    def __init__(self, num_nodes):
        self.graph = nx.DiGraph()
        self.nodes = list(range(num_nodes))
        self.graph.add_nodes_from(self.nodes)
        self.kira_node = None
        self.victims = set()
        self.latest_victim = None

        for n in self.nodes:
            self.graph.nodes[n]['suspicion_l'] = 0         # sospetto secondo L (A*)
            self.graph.nodes[n]['suspicion_n'] = 0         # sospetto secondo Near (Prolog)
            self.graph.nodes[n]['interactions'] = 0
            self.graph.nodes[n]['trust'] = rnd(0.85, 1.0)
            self.graph.nodes[n]['is_victim'] = False
            self.graph.nodes[n]['is_kira'] = False
            self.graph.nodes[n]['planted_evidence'] = False
            self.graph.nodes[n]['declarations'] = []


    def add_interaction(self, source, target):
        self.graph.add_edge(source, target)
        self.graph.nodes[source]['interactions'] += 1

    def mark_victim(self, node):
        self.graph.nodes[node]['is_victim'] = True
        self.victims.add(node)
        self.latest_victim = node

    def set_kira(self, node):
        self.kira_node = node
        self.graph.nodes[node]['is_kira'] = True

    def plant_evidence(self, target):
        self.graph.nodes[target]['planted_evidence'] = True
        self.graph.nodes[target]['suspicion_l'] += 1
        self.graph.nodes[target]['suspicion_n'] += 1


    def simulate_interactions(self):
        for source in self.nodes:
            if random.random() < 0.25:
                target = random.choice([n for n in self.nodes if n != source])
                self.add_interaction(source, target)

    def simulate_declarations(self):
        for node in self.nodes:
            self.graph.nodes[node]['declarations'].clear()
            for _, target in self.graph.out_edges(node):
                lie_chance = 1.0 - self.graph.nodes[node]['trust']
                if self.latest_victim is not None and target == self.latest_victim:
                    lie_chance += 0.3
                if random.random() > lie_chance:
                    self.graph.nodes[node]['declarations'].append((node, target))
                else:
                    fake_target = random.choice([n for n in self.nodes if n != node])
                    self.graph.nodes[node]['declarations'].append((node, fake_target))

# === Kira Agent ===
class Kira:
    def __init__(self, network):
        self.network = network
        self.node = None

    def assign(self, node):
        self.node = node
        self.network.set_kira(node)

    def act(self):
        #target = random.choice([n for n in self.network.nodes if n != self.node])
        #self.network.add_interaction(self.node, target)

        if random.random() < 0.5:
            victim = random.choice([n for n in self.network.nodes if n != self.node and not self.network.graph.nodes[n]['is_victim']])
            self.network.mark_victim(victim)
            self.network.add_interaction(self.node, victim)
            return victim

        if random.random() < 0.3:
            framed = random.choice([n for n in self.network.nodes if n != self.node])
            self.network.plant_evidence(framed)
            self.network.add_interaction(self.node, framed)
        return None

# === Detective L (A* Search-Based) ===
class DetectiveL:
    def __init__(self, network):
        self.network = network

    def heuristic(self, node):
        score = 0
        data = self.network.graph.nodes[node]

        if self.network.latest_victim and self.network.graph.has_edge(node, self.network.latest_victim):
            score += 6

        declared = {t for (_, t) in data['declarations']}
        actual = set(self.network.graph.successors(node))
        mismatch = declared.symmetric_difference(actual)
        score += len(mismatch) * 6

        if data['planted_evidence']:
            score += 12

        return score

    def analyze(self):
        suspects = []
        visited = set()
        pq = []

        for start in self.network.nodes:
            if self.network.graph.nodes[start]['is_victim']:
                continue
            heapq.heappush(pq, (0, start, []))

        while pq:
            cost, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            total_cost = cost + self.heuristic(current)
            self.network.graph.nodes[current]['suspicion_l'] += total_cost
            #suspects.append((current, f"A* score: {total_cost}"))
            if total_cost > 0:
                suspects.append((current, "Flagged by L as suspicious"))


            for neighbor in self.network.graph.successors(current):
                if neighbor not in visited:
                    heapq.heappush(pq, (total_cost, neighbor, path + [current]))

        return suspects

    def guess_kira(self):
        scores = [(n, self.network.graph.nodes[n]['suspicion_l']) for n in self.network.nodes if not self.network.graph.nodes[n]['is_victim']]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None

    def guess_confidence(self):
        scores = [(n, self.network.graph.nodes[n]['suspicion_l']) for n in self.network.nodes if not self.network.graph.nodes[n]['is_victim']]
        if not scores:
            return 0.0
        scores.sort(key=lambda x: x[1], reverse=True)
        top_score = scores[0][1]
        avg_score = sum(score for _, score in scores) / len(scores)
        if top_score == 0:
            return 0.0
        return min((top_score - avg_score) / top_score, 1.0)

# === Detective Near (Prolog-Based Advanced Reasoner) ===
class DetectiveNear:
    def __init__(self, network):
        self.network = network
        self.prolog = Prolog()
        self.prolog.consult("rules_en.pl")

    def sync_facts(self):
        self.prolog.retractall("interaction(_,_)" )
        self.prolog.retractall("victim(_)" )
        self.prolog.retractall("declaration(_,_)" )
        self.prolog.retractall("planted(_)" )

        for u, v in self.network.graph.edges():
            self.prolog.assertz(f"interaction({u},{v})")

        for node in self.network.nodes:
            data = self.network.graph.nodes[node]
            if data['is_victim']:
                self.prolog.assertz(f"victim({node})")
            if data['planted_evidence']:
                self.prolog.assertz(f"planted({node})")
            for dec in data['declarations']:
                self.prolog.assertz(f"declaration({dec[0]},{dec[1]})")

    def analyze(self):
        self.sync_facts()
        suspects = []

        rules = [
            ("lies", 3),
            ("fake_declaration", 2),
            ("multi_victim_contact", 2),
            ("suspicious_behavior", 4),
            ("silent_operator", 2),
        ]

        for rule_name, suspicion_points in rules:
            results = list(self.prolog.query(f"{rule_name}(X)"))
            for r in results:
                node = int(r["X"])
                self.network.graph.nodes[node]['suspicion_n'] += suspicion_points
                suspects.append((node, f"{rule_name.replace('_', ' ').capitalize()}"))

        return suspects

    def guess_kira(self):
        scores = [(n, self.network.graph.nodes[n]['suspicion_n']) for n in self.network.nodes if not self.network.graph.nodes[n]['is_victim']]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0] if scores else None

    def guess_confidence(self):
        scores = [(n, self.network.graph.nodes[n]['suspicion_n']) for n in self.network.nodes if not self.network.graph.nodes[n]['is_victim']]
        if not scores:
            return 0.0
        scores.sort(key=lambda x: x[1], reverse=True)
        top_score = scores[0][1]
        avg_score = sum(score for _, score in scores) / len(scores)
        if top_score == 0:
            return 0.0
        return min((top_score - avg_score) / top_score, 1.0)

def collapse_suspicions(entries):
    """
    entries: list of (node, reason)
    returns: dict {node: Counter of reasons}
    """
    grouped = defaultdict(list)
    for node, reason in entries:
        grouped[node].append(reason)

    collapsed = {}
    for node, reasons in grouped.items():
        reason_counts = Counter(reasons)
        collapsed[node] = reason_counts
    return collapsed

# === Visualization ===
def draw_graph(G, title="Graph", pos=None, small=False,key=None):
    #pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines')
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_color = [
        'darkred' if G.nodes[n].get('is_kira') else (
            'black' if G.nodes[n].get('is_victim') else (
                'orange' if G.nodes[n].get('planted_evidence') else 'lightblue'
            )
        ) for n in G.nodes()
    ]
    node_text = [
        f"Node {n}<br>"
        f"L Suspicion: {G.nodes[n].get('suspicion_l', 0)}<br>"
        f"Near Suspicion: {G.nodes[n].get('suspicion_n', 0)}"
        for n in G.nodes()
    ]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in G.nodes()],
        hovertext=node_text,
        marker=dict(color=node_color, size=20),
        textposition="top center"
    )
    
    size = 300 if small else None
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=title, margin=dict(t=20, b=20), height=size))
    st.plotly_chart(fig, use_container_width=True,key = key)

def draw_suspicion_histogram(graph, title="Suspicion Histogram"):
    node_ids = list(graph.nodes())
    suspicion_l = [graph.nodes[n].get('suspicion_l', 0) for n in node_ids]
    suspicion_n = [graph.nodes[n].get('suspicion_n', 0) for n in node_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=node_ids, y=suspicion_l, name='Detective L (A*)', marker_color='skyblue'))
    fig.add_trace(go.Bar(x=node_ids, y=suspicion_n, name='Detective Near (Prolog)', marker_color='indianred'))

    fig.update_layout(
        barmode='group',
        title=title,
        xaxis_title="Node ID",
        yaxis_title="Suspicion Level",
        height=400
    )
    fig.update_layout(xaxis=dict(type='category'))
    return fig


# === Main Execution ===
def run_sim():
    st.set_page_config(layout="wide")
    col0, col00 = st.columns([0.7, 1], gap="large")

    with col0:
        st.title("🕵️‍♂️ AI Simulation: Kira in the Network")
        num_nodes = st.slider("Number of people in the network", 5, 20, 10)
        num_turns = st.slider("Simulation turns", 1, 10, 5)

    with col00:
        st.title("Description")
        st.markdown("""
Welcome to **Kira: AI Simulation**, an interactive investigative experiment inspired by *Death Note*, designed to test intelligent agents within a complex and dynamic social network.

In this simulation:
- A network of individuals interacts over time.
- Among them hides **Kira**, a strategic killer capable of murdering, deceiving, and planting false evidence.
- Two AI detectives, **L** and **Near**, analyze behaviors to uncover the truth:
    - 🕵️‍♂️ **Detective L** uses a search algorithm inspired by A*, guided by heuristic reasoning based on network structure.
    - 🧠 **Detective Near** applies symbolic logic through a Prolog rule engine to detect contradictions and suspicious patterns.

Each turn simulates:
- Social interactions (random or planned),
- Possible murders by Kira,
- Declarations (truthful or deceptive),
- Analysis by the detectives and suspicion updates.

The ultimate goal is simple:
**Unmask Kira**... before it's too late.

This simulation is designed to be:
- 👁️‍🗨️ **Interactive and transparent**, featuring network graphs, logbooks, and suspicion histograms,
- 🧪 **Educational**, showcasing different AI techniques for social reasoning and deductive logic,
- 🎮 **Modular and expandable**, ready to support new agents, rules, or advanced mechanics.

Are you ready to watch artificial intelligence... solve a murder?
""")

    if st.button("Start Simulation"):
        network = SocialNetwork(num_nodes)
        fixed_positions = nx.spring_layout(network.graph, seed=42)
        kira = Kira(network)
        kira.assign(random.choice(network.nodes))
        detective_l = DetectiveL(network)
        near = DetectiveNear(network)
        logbook = []
        logbook_n = []

        snapshots = []

        for turn in range(num_turns):
            network.simulate_interactions()
            victim = kira.act()
            network.simulate_declarations()
            suspects = detective_l.analyze()
            suspects_n = near.analyze()
            if victim is not None:
                logbook.append((turn, victim, "Was killed this turn"))
                logbook_n.append((turn, victim, "Was killed this turn"))
            for s in suspects:
                logbook.append((turn, s[0], s[1]))
            for s in suspects_n:
                logbook_n.append((turn, s[0], s[1]))
            snapshots.append(copy.deepcopy(network.graph))

        st.success(f"Simulation completed. Kira was node {kira.node}.")

        st.divider()

        st.markdown("### 🧬 Complete Network Graph")
        legend_text = """
        **Legend**:
        - 🔴 Red: Kira
        - ⚫ Black: Killed
        - 🟠 Orange: Framed
        - 🔵 Blue: Innocent
        """
        st.markdown(legend_text)

        fixed_positions = nx.spring_layout(network.graph, seed=42,k=0.8, iterations=100)
        colA, colB = st.columns([1.4,1], gap="medium")
        with colA:
            draw_graph(network.graph, title="Final State Network", pos=fixed_positions)
        with colB:
            st.plotly_chart(draw_suspicion_histogram(network.graph))

        st.markdown("### 🎯 Final AI Guesses")

        col_l, col_n = st.columns(2)

        with col_l:
            st.markdown("#### 🕵️ Detective L (A*)")
            guess = detective_l.guess_kira()
            confidence = detective_l.guess_confidence()
            st.markdown(f"Suspects node **{guess}**")
            if guess == kira.node:
                st.success("Correct! 🎯")
            else:
                st.error("Incorrect ❌")
            st.progress(confidence, text=f"Confidence: {int(confidence * 100)}%")

        with col_n:
            st.markdown("#### 🧠 Detective Near (Prolog)")
            guess_n = near.guess_kira()
            confidence_n = near.guess_confidence()
            st.markdown(f"Suspects node **{guess_n}**")
            if guess_n == kira.node:
                st.success("Correct! 🎯")
            else:
                st.error("Incorrect ❌")
            st.progress(confidence_n, text=f"Confidence: {int(confidence_n * 100)}%")

        # Group logbook by turn (if not already)
        grouped_log_l = {}
        grouped_log_n = {}
        for entry in logbook:
            turn, node, reason = entry
            if "killed" not in reason.lower():
                grouped_log_l.setdefault(turn, []).append((node, reason))
        for entry in logbook_n:
            turn, node, reason = entry
            grouped_log_n.setdefault(turn, []).append((node, reason))

        # Build UI: One expander per turn with 3 columns
        for i, graph in enumerate(snapshots):
            with st.expander(f"Turn {i}"):
                col1, col2, col3, col4 = st.columns([1.3,0.8,0.6,0.6], gap="small")

                with col1:
                    #st.markdown("#### 🔁 Network Graph")
                    draw_graph(graph, title=f"Network Graph", small=True,key=f"graph_{i}", pos=fixed_positions)
                
                with col2:
                    #st.markdown("#### 📊 Suspicion Histogram")
                    st.plotly_chart(draw_suspicion_histogram(graph), use_container_width=True, key=f"suspicion_hist_{i}")


                with col3:
                    st.markdown("#### 📜 Detective L's Logbook")
                    suspects_l= grouped_log_l.get(i, [])
                    if suspects_l:
                        for node,reason in suspects_l:
                            st.markdown(f"- Node {node}: _{reason}_")
                    else:
                        st.markdown("No suspects this turn.")

                with col4:
                    st.markdown("#### 🧠 Detective Near's Logbook")
                    suspects_n = grouped_log_n.get(i,[])

                    collapsed = collapse_suspicions(suspects_n)
                    if suspects_n:
                        for node, reasons in collapsed.items():
                            reason_text = ', '.join(
                                f"{reason} (×{count})" if count > 1 else reason
                                for reason, count in reasons.items()
                            )
                            st.markdown(f"- Node {node}: _{reason_text}_")
                    else: 
                        st.markdown("No suspects this turn.")
        st.divider()
    st.markdown("### 🧪 Batch Testing (opzionale)")
   

    if st.button("Esegui 30 simulazioni di test"):
        batch_test_ui(num_tests=30, num_nodes=num_nodes, num_turns=num_turns)
        st.markdown("#### ✅ Risultati:")
  


def simulate_once(num_nodes=10, num_turns=5, verbose=False):
    network = SocialNetwork(num_nodes)
    kira = Kira(network)
    kira.assign(random.choice(network.nodes))
    detective_l = DetectiveL(network)
    near = DetectiveNear(network)

    for _ in range(num_turns):
        network.simulate_interactions()
        kira.act()
        network.simulate_declarations()
        detective_l.analyze()
        near.analyze()

    guess_l = detective_l.guess_kira()
    guess_n = near.guess_kira()
    conf_l = detective_l.guess_confidence()
    conf_n = near.guess_confidence()

    result = {
        "kira": kira.node,
        "guess_l": guess_l,
        "guess_n": guess_n,
        "correct_l": guess_l == kira.node,
        "correct_n": guess_n == kira.node,
        "confidence_l": conf_l,
        "confidence_n": conf_n,
    }

    if verbose:
        print(result)

    return result

def run_batch_test(num_tests=30, num_nodes=10, num_turns=5):
    results = [simulate_once(num_nodes, num_turns) for _ in range(num_tests)]

    correct_l = sum(r["correct_l"] for r in results)
    correct_n = sum(r["correct_n"] for r in results)
    avg_conf_l = sum(r["confidence_l"] for r in results) / num_tests
    avg_conf_n = sum(r["confidence_n"] for r in results) / num_tests

    print(f"\nBatch results on {num_tests} simulations:")
    print(f"Detective L: correct {correct_l}/{num_tests} ({correct_l/num_tests:.2%}), avg confidence: {avg_conf_l:.2f}")
    print(f"Detective Near: correct {correct_n}/{num_tests} ({correct_n/num_tests:.2%}), avg confidence: {avg_conf_n:.2f}")


def batch_test_ui(num_tests, num_nodes, num_turns):
    results = [simulate_once(num_nodes, num_turns) for _ in range(num_tests)]
    correct_l = sum(r["correct_l"] for r in results)
    correct_n = sum(r["correct_n"] for r in results)
    avg_conf_l = sum(r["confidence_l"] for r in results) / num_tests
    avg_conf_n = sum(r["confidence_n"] for r in results) / num_tests

    st.markdown("#### ✅ Risultati:")
    st.markdown(f"- **Detective L:** {correct_l}/{num_tests} corrette ({correct_l/num_tests:.1%}), confidenza media: {avg_conf_l:.2f}")
    st.markdown(f"- **Detective Near:** {correct_n}/{num_tests} corrette ({correct_n/num_tests:.1%}), confidenza media: {avg_conf_n:.2f}")

    df = pd.DataFrame({
        "Detective": ["Detective L", "Detective Near"],
        "Corrette": [correct_l, correct_n],
        "Confidenza media": [avg_conf_l, avg_conf_n]
    })

    st.markdown("#### 📊 Confronto Visivo")
    fig = px.bar(df, x="Detective", y="Corrette", color="Detective",
                 text="Corrette", barmode="group", height=400)
    fig.update_layout(showlegend=False, yaxis_title="Predizioni Corrette", xaxis_title="")
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    run_sim()
