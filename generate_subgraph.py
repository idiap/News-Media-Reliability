import json
import networkx as nx

from pyvis.network import Network
from coloraide import Color


color_unknown = "#bdbdbd"
color_reliable = "#58D75B"
color_unreliable = "#e53935"
color_unknown_edge = "#979797"
color_reliable_edge = "#3F9A40"
color_unreliable_edge = "#A42825"
node_min_size = 10
node_max_size = 100
edge_max_width = 40
edge_min_width = 4
edge_min_fr = 50

G = None  # Global graph
domain2node = {}
node2domain = {}


def load_global_graph(path_graph, path_index, path_reliability_scores):
    global G, domain2node, node2domain

    # Loading domian to node indices conversion
    with open(path_index) as reader:
        index = reader.read().split("\n")
    for line in index:
        if not line:
            continue
        domain, node = line.split("\t")
        domain2node[domain] = int(node)
        node2domain[int(node)] = domain

    # Loading graph and removing infrequent edges
    G = nx.read_graphml(path_graph, node_type=int)
    G.remove_edges_from([(a, b) for a, b, fr in G.edges(data="fr") if fr <= edge_min_fr or a == b])
    G.remove_nodes_from(list(nx.isolates(G)))

    # Loading reliability values
    with open(path_reliability_scores) as reader:
        news_sources_info = json.load(reader)

    # Visualization configuration
    color_interpolation_pos = Color.interpolate([color_unknown, color_reliable], space='lch')
    color_interpolation_neg = Color.interpolate([color_unknown, color_unreliable], space='lch')
    color_interpolation_pos_edge = Color.interpolate([color_unknown_edge, color_reliable_edge], space='lch')
    color_interpolation_neg_edge = Color.interpolate([color_unknown_edge, color_unreliable_edge], space='lch')
    reliability_method = "p+fp-average"  # "p-reliability"
    max_weights = {}
    for node in G.nodes:
        if G.out_edges(node, data=True):
            max_weights[node] = max(d["weight"] for _, _, d in G.out_edges(node, data=True))
    for s, t, d in G.edges(data=True):
        d['prob'] = d['weight']
        out_norm_weight = d['prob'] / max_weights[s]
        d['weight'] = max(edge_max_width * out_norm_weight, edge_min_width)
        reliability = news_sources_info[node2domain[s]][reliability_method]
        if reliability >= 0:
            d["color"] = color_interpolation_pos_edge(reliability).to_string(hex=True)
        else:
            d["color"] = color_interpolation_neg_edge(-reliability).to_string(hex=True)
    for id in G.nodes:
        node = G.nodes[id]
        node["label"] = node2domain[id]
        node["reliability"] = news_sources_info[node2domain[id]][reliability_method]
        node["size"] = node_min_size + int(abs(node["reliability"]) * node_max_size)
        if node["reliability"] >= 0:
            node["color"] = color_interpolation_pos(node["reliability"]).to_string(hex=True)
        else:
            node["color"] = color_interpolation_neg(-node["reliability"]).to_string(hex=True)
        node["font"]={"size": 150,
                      "color": "black",
                      "strokeColor": "black",
                      "strokeWidth": 0}


def get_subgraph_html(target_node, graph_type_edges="in", graph_max_edges=150):
    if target_node not in domain2node:
        if not target_node.startswith("www.") and "www." + target_node in domain2node:
            target_node = "www." + target_node
        elif target_node.startswith("www.") and target_node[4:] in domain2node: 
            target_node = target_node[4:]
        else:
            return ""

    target_node = domain2node[target_node]
    G_sub = nx.DiGraph()
    if graph_type_edges != "out":
        G_sub.add_edges_from(G.in_edges(target_node, data=True))
    if graph_type_edges != "in":
        G_sub.add_edges_from(G.out_edges(target_node, data=True))
    nx.set_node_attributes(G_sub, {n:G.nodes[n] for n in G_sub.nodes})
    G_sub.nodes[target_node]["font"]["strokeWidth"] = 10

    if len(G_sub.edges) > graph_max_edges:
        if len(G_sub.edges) > graph_max_edges:
            remove_edges = [(a, b)
                    for a, b, _
                    in  sorted([(a, b, w) for a, b, w in G_sub.edges(data="weight")],
                                key=lambda abw: -abw[-1])[graph_max_edges:]]
        G_sub.remove_edges_from(remove_edges)
        G_sub.remove_nodes_from(list(nx.isolates(G_sub)))

    nt = Network(height="750px", width="100%", directed=True)
    nt.from_nx(G_sub)
    nt.barnes_hut()

    # nt.write_html(path_html_out)
    html = nt.generate_html()
    onload_str = "// really clean the dom element"
    onload_ix = html.find(onload_str)
    if onload_ix == -1:
        onload_str = "return network;"
        onload_ix = html.find(onload_str)
    # forcing full screen mode and disabling physics onload
    html = html[:onload_ix] + \
            "document.getElementById('mynetwork').style.height = '100%';" + \
            "network.setOptions({height: window.innerHeight + 'px'});" + \
            html[onload_ix:]
    # html = html[:onload_ix] + \
    #         "document.getElementById('mynetwork').style.height = '100%';" + \
    #         "network.setOptions({height: window.innerHeight + 'px'});" + \
    #         "network.setOptions({physics : false});\n" + \
            # html[onload_ix:]
    return html

if __name__ == "__main__":
    import sys
    import os

    GRAPH_PATH = "CC-NEWS/ALL/08/graph/en/"
    load_global_graph(
        os.path.join(GRAPH_PATH, "graph_news.graphml"),
        os.path.join(GRAPH_PATH, "domain_ix_news.tsv"),
        os.path.join(GRAPH_PATH, "reliability_scores.json")
    )

    target = "wordpress.com" if len(sys.argv) <= 1 else sys.argv[1]
    html = get_subgraph_html(target, graph_type_edges="in")  # graph_type_edges is either: "in", "out", or "both"
    if html:
        with open(os.path.join(GRAPH_PATH, f"subgraph-{target}.html"), "w+") as out:
            out.write(html)
    else:
        print(f"target url {target} is not found in the graph")
