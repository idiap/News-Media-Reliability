# -*- coding: utf-8 -*-
"""
Script to construct and evaluate the news media sources graph with reliability value computation

Copyright (c) 2024 Idiap Research Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import gzip
import json
import hydra
import logging
import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from random import randint
from omegaconf import DictConfig
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tools.util import investment, value_iteration, reverse_bellman_iteration

SEED = 13
K_FOLD = 5

Log = None


def print_topn_sources(V:np.ndarray, ix2domain:dict, top_n:int=10) -> None:
    news_sources = V.argsort()

    print(f"\nTop-{top_n} most reliable sources:")
    for ix in range(1, top_n + 1):
        s = news_sources[-ix]
        print(f"  {ix}. {ix2domain[s]} ({V[s]:.4f})")

    print(f"\nTop-{top_n} least reliable sources:")
    for ix in range(1, top_n + 1):
        s = news_sources[ix - 1]
        print(f"  {ix}. {ix2domain[s]} ({V[s]:.4f})")


def rank_and_normalize(scores, method):
    scores = [e for e in scores.values()]
    scores.sort(key=lambda e: -e[method])
    rank = -1
    last_score = None
    for e in scores:
        if e[method] < 0:
            break
        if e[method] != last_score:
            rank += 1
            last_score = e[method]
        e[f'{method}_rank_norm'] = rank
    for e in scores:
        if e[method] < 0:
            break
        e[f'{method}_rank_norm'] = 1 - e[f'{method}_rank_norm'] / rank
    scores.sort(key=lambda e: e[method])
    rank = 1
    last_score = None
    for e in scores:
        if e[method] >= 0:
            break
        if e[method] != last_score:
            rank -= 1
            last_score = e[method]
        e[f'{method}_rank_norm'] = rank
    for e in scores:
        if e[method] >= 0:
            break
        e[f'{method}_rank_norm'] = -(1 - e[f'{method}_rank_norm'] / (rank - 1))

    for e in scores:
        e[method] = e[f'{method}_rank_norm']
        del e[f'{method}_rank_norm']


def norm_neg_pos_vec(V):
    V[V > 0] /= V.max()
    V[V < 0] /= -V.min()
    return V


def save_indexes(mapping:dict, only_news:bool, output_path:str) -> None:
    from tools.util import GzippingWrapper

    output_file = os.path.join(output_path, f"domain_ix_{'news' if only_news else 'all'}.gz")
    output_file = GzippingWrapper(open(output_file, "wb"))
    for domain in mapping:
        output_file.write(f"{domain}\t{mapping[domain]}\n".encode('utf-8'))
    output_file.close()


def create_indexes(cfg:DictConfig, only_news:bool, news_sources_files:str) -> dict:
    domain2ix = {}

    for file_path in tqdm(news_sources_files, desc="Creating indexes"):
        with gzip.open(file_path, 'rb') as reader:
            domains = json.load(reader)
            for domain in domains:
                if only_news and domain in cfg.graph.news_sources.ignore:
                    continue

                if domain not in domain2ix:
                    domain2ix[domain] = len(domain2ix)

                if not only_news:
                    for link_domain in domains[domain]:
                        if link_domain not in domain2ix:
                            domain2ix[link_domain] = len(domain2ix)

    return domain2ix


def create_and_save_graph(cfg:DictConfig, domain2ix:dict, graph_type:str,
                          news_sources_files:str, output_path:str) -> nx.DiGraph:

    G = nx.DiGraph()

    filter_nodes = []
    cfg_filter = cfg.graph.news_sources.filter
    if cfg_filter.targets:
        filter_nodes = [domain2ix[n] for n in cfg_filter.targets if n in domain2ix]

    ix2domain = {}
    for domain in domain2ix:
        ix2domain[domain2ix[domain]] = domain

    for file_path in tqdm(news_sources_files, desc="Graph creation"):
        with gzip.open(file_path, 'rb') as reader:
            domains = json.load(reader)
            for domain in domains:
                if domain not in domain2ix:
                    continue

                source_node = domain2ix[domain]
                for link_domain in domains[domain]:
                    if link_domain in domain2ix and link_domain != domain:  # removing self-loops
                        target_node = domain2ix[link_domain]
                        if cfg_filter.targets and not cfg_filter.include_neighbors and\
                        (source_node not in filter_nodes or target_node not in filter_nodes):
                            continue

                        edge = G.get_edge_data(source_node, target_node)
                        if not edge:
                            G.add_edge(source_node, target_node, fr=domains[domain][link_domain])
                        else:
                            edge['fr'] += domains[domain][link_domain]

    edges_to_remove = []
    for domain_ix in tqdm(G.nodes, desc="Graph edge weight"):
        out_edges = G.out_edges(domain_ix, data=True)
        total_fr = sum([d['fr'] for _, _, d in out_edges])
        for a, b, d in out_edges:
            if cfg_filter.targets and a not in filter_nodes and b not in filter_nodes:
                edges_to_remove.append((a, b))
            d['weight'] = d['fr'] / total_fr

        in_edges = G.in_edges(domain_ix, data=True)
        total_fr = sum([d['fr'] for _, _, d in in_edges])
        for a, b, d in in_edges:
            d['weight_in'] = d['fr'] / total_fr
            d['weight_in'] = d['fr'] / total_fr

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    # nx.write_gpickle(G, os.path.join(output_path, f"graph_{graph_type}.pkl"))

    print(f"Graph created ({len(G.nodes)} nodes, {len(G.edges)} edges)")

    golden_truth_file = os.path.join(cfg.golden_truth.output_path, cfg.golden_truth.output_file)
    if not os.path.exists(golden_truth_file):
        Log.warning("File with news sources reliability values (golden truth) not found. Trying creating it...")
        os.system("python build_dataset.py")

    print("Reading news sources reliability values (golden truth) file...")
    df_dataset = pd.read_csv(golden_truth_file)

    df_dataset.reliability_label = df_dataset.reliability_label.map(lambda v: -1 if v < -1 else v)
    if cfg.golden_truth.include_mixed_in_reward:
        df_dataset.reliability_label = df_dataset.reliability_label.map(lambda v: -1 if v == 0 else v)

    print("  Grund truth total values:", len(df_dataset))
    print("  Grund truth total values in the Graph:", len(df_dataset[df_dataset.domain.isin(domain2ix)]))
    print("  Grund truth total values in the Graph label distribution:")
    print(df_dataset[df_dataset.domain.isin(domain2ix)].reliability_label.value_counts())

    news_sources_reliability_info = {domain2ix[r.domain]:r.reliability_label
                                     for _, r in df_dataset.iterrows()
                                     if r.domain in domain2ix}
    ground_truth = news_sources_reliability_info.copy()

    rewards = news_sources_reliability_info.copy()

    V_neg = value_iteration(G, rewards, only_neg_rewards=True, gamma=0.05)
    V_pos = reverse_bellman_iteration(G, rewards, only_pos_rewards=True, gamma=0.05)
    V_ri = V_neg + V_pos

    V_rbi = reverse_bellman_iteration(G, rewards, only_pos_rewards=False, gamma=0.1)

    V_vi = value_iteration(G, rewards, gamma=0.05)

    V_invest = investment(G, rewards, max_iter=1)

    news_sources_info = {}
    news_pagerank = nx.pagerank(G)
    max_pagerank = max(news_pagerank.values())

    V_ri = norm_neg_pos_vec(V_ri)
    V_vi = norm_neg_pos_vec(V_vi)

    V_rbi = norm_neg_pos_vec(V_rbi)
    V_invest = norm_neg_pos_vec(V_invest)

    for node_id in news_pagerank:
        news_sources_info[node_id] = {
            "f-reliability": V_vi[node_id],
            "p-reliability": V_rbi[node_id],
            "fp-reliability": V_ri[node_id],
            "i-reliability": V_invest[node_id],
            "pagerank": news_pagerank[node_id] / max_pagerank,
            "ground_truth": ground_truth[node_id] if node_id in ground_truth else None
        }

    for method in next(iter(news_sources_info.values())):
        if method != "ground_truth":
            rank_and_normalize(news_sources_info, method)
    for domain in news_sources_info:
        news_sources_info[domain]["p+fp-average"] = (news_sources_info[domain]["p-reliability"] + news_sources_info[domain]["fp-reliability"]) / 2

    with open(os.path.join(output_path, "reliability_scores.json"), "w") as writer:
        news_sources_info_json = {}
        for node_id in news_sources_info:
            news_sources_info_json[ix2domain[node_id]] = news_sources_info[node_id]
        json.dump(news_sources_info_json, writer)
        del news_sources_info_json

    if cfg.graph.evaluation:
        X = [k for k in news_sources_reliability_info.keys()
             if k in ix2domain and (cfg.golden_truth.include_mixed_in_reward or news_sources_reliability_info[k] != 0)]
        y = [int(ground_truth[k] > 0) for k in X]

        X = np.array(X)
        y = np.array(y)

        print("Dataset size:", len(y))
        unique, counts = np.unique(y, return_counts=True)
        print("Label distribution:")
        print(f"  - '{'reliable' if unique[0] > 0 else 'not reliable'}': {counts[0]}")
        print(f"  - '{'reliable' if unique[1] > 0 else 'not reliable'}': {counts[1]}")

        pbar_desc = f"{K_FOLD}-fold"
        progress_bar = tqdm(total=K_FOLD, desc=pbar_desc)
        reports = defaultdict(lambda:[None] * K_FOLD)
        skf = StratifiedKFold(n_splits=K_FOLD, random_state=SEED, shuffle=True)
        for i_fold, (_, test_ix) in enumerate(skf.split(X, y)):
            x_test, y_test = X[test_ix], y[test_ix]

            rewards = news_sources_reliability_info.copy()
            for k in x_test:
                rewards[k] = 0

            reports["Random Baseline"][i_fold] = classification_report(y_test, [randint(0, 1) for _ in x_test], output_dict=True, zero_division=0)
            reports["Majority Baseline"][i_fold] = classification_report(y_test, [1 for _ in x_test], output_dict=True, zero_division=0)

            V = value_iteration(G, rewards, verbose=False, gamma=0.05)
            y_pred = [int(V[k] > 0) for k in x_test]
            reports["F-Reliability (gamma=0.05)"][i_fold] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            V = reverse_bellman_iteration(G, rewards, only_pos_rewards=False, verbose=False, gamma=0.3)
            y_pred = [int(V[k] > 0) for k in x_test]
            reports["P-Reliability (gamma=0.3)"][i_fold] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            V_neg = value_iteration(G, rewards, only_neg_rewards=True, verbose=False, gamma=0.05)
            V_pos = reverse_bellman_iteration(G, rewards, only_pos_rewards=True, verbose=False, gamma=0.05)
            V = V_neg + V_pos
            y_pred = [int(V[k] > 0) for k in x_test]
            reports["FP-Reliability (gamma=0.05)"][i_fold] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            V_invest = investment(G, rewards, max_iter=1)
            y_pred = [int(V_invest[k] > 0) for k in x_test]
            reports["I-Reliability (n=1)"][i_fold] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            progress_bar.update(1)
        progress_bar.set_description_str(pbar_desc + " [finished]")
        progress_bar.close()

        # printing results
        print()
        results_all = {}
        for method in reports:
            print(f"> {method}:")
            f1_folds = np.array([reports[method][i]["macro avg"]["f1-score"] for i in range(K_FOLD)])
            f1_pos_folds = np.array([reports[method][i]["1"]["f1-score"] for i in range(K_FOLD)])
            f1_neg_folds = np.array([reports[method][i]["0"]["f1-score"] for i in range(K_FOLD)])

            precision_folds = np.array([reports[method][i]["macro avg"]["precision"] for i in range(K_FOLD)])
            precision_pos_folds = np.array([reports[method][i]["1"]["precision"] for i in range(K_FOLD)])
            precision_neg_folds = np.array([reports[method][i]["0"]["precision"] for i in range(K_FOLD)])

            recall_folds = np.array([reports[method][i]["macro avg"]["recall"] for i in range(K_FOLD)])
            recall_pos_folds = np.array([reports[method][i]["1"]["recall"] for i in range(K_FOLD)])
            recall_neg_folds = np.array([reports[method][i]["0"]["recall"] for i in range(K_FOLD)])

            accuracy_folds = np.array([reports[method][i]["accuracy"] for i in range(K_FOLD)])

            results_all[method] = {
                "f1": (f1_folds * 100).tolist(),
                "f1_pos": (f1_pos_folds * 100).tolist(),
                "f1_neg": (f1_neg_folds * 100).tolist(),
                "precision": (precision_folds * 100).tolist(),
                "precision_pos": (precision_pos_folds * 100).tolist(),
                "precision_neg": (precision_neg_folds * 100).tolist(),
                "recall": (recall_folds * 100).tolist(),
                "recall_pos": (recall_pos_folds * 100).tolist(),
                "recall_neg": (recall_neg_folds * 100).tolist(),
                "accuracy": (accuracy_folds * 100).tolist()
            }
            results = {"f1": (f1_folds.mean(), f1_folds.std()),
                    "f1_pos": (f1_pos_folds.mean(), f1_pos_folds.std()),
                    "f1_neg": (f1_neg_folds.mean(), f1_neg_folds.std()),
                    "precision": (precision_folds.mean(), precision_folds.std()),
                    "precision_pos": (precision_pos_folds.mean(), precision_pos_folds.std()),
                    "precision_neg": (precision_neg_folds.mean(), precision_neg_folds.std()),
                    "recall": (recall_folds.mean(), recall_folds.std()),
                    "recall_pos": (recall_pos_folds.mean(), recall_pos_folds.std()),
                    "recall_neg": (recall_neg_folds.mean(), recall_neg_folds.std()),
                    "accuracy": (accuracy_folds.mean(), accuracy_folds.std())}
            results = {k:(m * 100, s * 100) for (k, (m, s)) in results.items()}

            print(f"  F1 Macro: {results['f1'][0]:.2f} +/- {results['f1'][1]:.2f}")
            print(f"  F1 Reliable: {results['f1_pos'][0]:.2f} +/- {results['f1_pos'][1]:.2f}")
            print(f"  F1 Unreliable: {results['f1_neg'][0]:.2f} +/- {results['f1_neg'][1]:.2f}")
            # print(f"  Precision Macro: {results['precision'][0]:.2f} +/- {results['precision'][1]:.2f}")
            # print(f"  Precision Reliable: {results['precision_pos'][0]:.2f} +/- {results['precision_pos'][1]:.2f}")
            # print(f"  Precision Unreliable: {results['precision_neg'][0]:.2f} +/- {results['precision_neg'][1]:.2f}")
            # print(f"  Recall Macro: {results['recall'][0]:.2f} +/- {results['recall'][1]:.2f}")
            # print(f"  Recall Reliable: {results['recall_pos'][0]:.2f} +/- {results['recall_pos'][1]:.2f}")
            # print(f"  Recall Unreliable: {results['recall_neg'][0]:.2f} +/- {results['recall_neg'][1]:.2f}")
            print(f"  Accuracy: {results['accuracy'][0]:.2f} +/- {results['accuracy'][1]:.2f}")
            print(f"  ----")

            # Print Table row in LaTeX:
            # row = ""
            # for metric in ["precision", "recall", "f1"]:
            #     for target in ["", "_pos", "_neg"]:
            #         row += f" & {results[metric+target][0]:.2f}$\pm${results[metric+target][1]:.2f}"
            # row += f" & {results['accuracy'][0]:.2f}$\pm${results['accuracy'][1]:.2f}" + r" \\"
            # print(row)

    cfg_v = cfg.graph.visualization
    if cfg_v.enable:
        from pyvis.network import Network
        from coloraide import Color

        color_interpolation_pos = Color.interpolate([cfg_v.color.unknown, cfg_v.color.reliable], space='lch')
        color_interpolation_neg = Color.interpolate([cfg_v.color.unknown, cfg_v.color.unreliable], space='lch')

        print("  [visualization] color and size computation for edges...")
        reliability_method = "p+fp-average"  # "p-reliability"
        for s, t, d in G.edges(data=True):
            d['prob'] = d['weight']
            d['weight'] = max(cfg_v.graph.edge.max_width * d['weight'], cfg_v.graph.edge.min_width)
            reliability = news_sources_info[s][reliability_method]

            if not cfg_filter.include_neighbors or s not in filter_nodes:
                if reliability >= 0:
                    d["color"] = color_interpolation_pos(reliability).to_string(hex=True)
                else:
                    d["color"] = color_interpolation_neg(-reliability).to_string(hex=True)
            else:
                d["color"] = cfg_filter.color

            if s in news_sources_reliability_info:
                if news_sources_reliability_info[s] > 0:
                    d["color"] = cfg_v.color.reliable
            if t in news_sources_reliability_info:
                if news_sources_reliability_info[t] < 0:
                    d["color"] = cfg_v.color.unreliable

        print("  [visualization] color and size computation for nodes...")
        for id in G.nodes:
            node = G.nodes[id]

            self_link = G.get_edge_data(id, id)
            if self_link:
                node["prob"] = self_link['prob']
            else:
                node["prob"] = 0

            node["label"] = ix2domain[id]
            node["reliability"] = news_sources_info[id][reliability_method]
            node["size"] = cfg_v.graph.node.min_size + int(abs(node["reliability"]) * cfg_v.graph.node.max_size)
            if node["reliability"] >= 0:
                node["color"] = color_interpolation_pos(node["reliability"]).to_string(hex=True)
            else:
                node["color"] = color_interpolation_neg(-node["reliability"]).to_string(hex=True)

            if id in news_sources_reliability_info:
                if news_sources_reliability_info[id] > 0:
                    node["color"] = cfg_v.color.reliable
                elif news_sources_reliability_info[id] < 0:
                    node["color"] = cfg_v.color.unreliable

        print("  [visualization] removing edges and isolated nodes...")
        edges_to_remove = [(a,b) for a, b, attrs in G.edges(data=True) if attrs["fr"] <= cfg_v.graph.edge.min_fr or a == b]
        G.remove_edges_from(edges_to_remove)
        if filter_nodes:
            isolated_nodes = [n for n in nx.isolates(G) if n not in filter_nodes]
        else:
            isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)

        html_path = os.path.join(output_path, f"graph_{graph_type}.html")
        print(f"  [visualization] creating HTML file with visualization ({html_path})...")
        nt = Network(height="750px", width="100%", directed=True,
                    select_menu=False, filter_menu=True,
                    bgcolor="#222222", font_color="white")
        nt.from_nx(G)
        nt.barnes_hut()
        nt.toggle_hide_edges_on_drag(True)
        nt.inherit_edge_colors(False)

        neighbor_map = nt.get_adj_list()
        # add neighbor data to node hover data
        for node in nt.nodes:

            if cfg_filter.include_neighbors and node['id'] in filter_nodes:
                node["color"] = {"background": node["color"], "border": cfg_filter.color}
                node["borderWidth"] = 10

            neighbors = [('self', node['prob'])] + [(nt.get_node(d)['label'], G.get_edge_data(node['id'], d)['prob'])
                                                    for d in neighbor_map[node["id"]]]
            neighbors = sorted(neighbors, key=lambda n: -n[1])
            neighbors = [f"{ix + 1}. {dp[0]} ({dp[1]:.2%})" for ix, dp in enumerate(neighbors)]
            node["title"] = f"{node['label']}\n" + \
                            f"(reliability degree: {news_sources_info[node['id']][reliability_method]:.2%})" + \
                            (f"\n\nLinks to:\n" + "\n  ".join(neighbors) if neighbors else '')
            del node["reliability"], node['prob']

        # nt.toggle_disable_physics_onload(True)  # https://github.com/WestHealth/pyvis/pull/179
        # nt.show(html_path)
        nt.write_html(html_path)
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
               "network.setOptions({physics : false});\n" + \
               html[onload_ix:]
        with open(html_path, "w+") as out:
            out.write(html)
        os.system(f"rm -f -r {output_path}/lib")
        os.system(f"mv lib {output_path}")

    return G


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg:DictConfig) -> None:
    global Log

    logging.basicConfig(level=logging.INFO)
    Log = logging.getLogger(__name__)

    path_ccnews_snapshot_cache = cfg.snapshot.paths_url.split(cfg.snapshot.url_split)[1]
    path_ccnews_snapshot_cache = os.path.split(path_ccnews_snapshot_cache)[0]
    path_target_folder = os.path.join(path_ccnews_snapshot_cache, cfg.output.news_sources_folder, cfg.graph.lang)

    if not cfg.graph.join_graphs and not os.path.exists(path_target_folder):
        Log.critical(f"wrong target language '{cfg.graph.lang}' and/or CC News snapshot '{cfg.snapshot.paths_url}'")
        exit()

    if cfg.graph.join_graphs:
        news_sources_files = [os.path.join(f"CC-NEWS/{year}/{month}/{cfg.output.news_sources_folder}/{cfg.graph.lang}", file)
                              for year in os.listdir("CC-NEWS")
                              for month in os.listdir(f"CC-NEWS/{year}")
                              if os.path.exists(f"CC-NEWS/{year}/{month}/{cfg.output.news_sources_folder}/{cfg.graph.lang}")
                              for file in os.listdir(f"CC-NEWS/{year}/{month}/{cfg.output.news_sources_folder}/{cfg.graph.lang}")
                              if year.isdigit()]
        output_path = os.path.join(re.sub(r"\d{4}", "ALL", path_ccnews_snapshot_cache), cfg.output.graph_folder, cfg.graph.lang)
    else:
        news_sources_files = [os.path.join(path_target_folder, f) for f in os.listdir(path_target_folder)]
        output_path = os.path.join(path_ccnews_snapshot_cache, cfg.output.graph_folder, cfg.graph.lang)

    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    for graph_type in cfg.graph.target_graph:
        print(f"\nGraph '{graph_type}'")
        is_only_news = graph_type == 'news'
        domain2ix = create_indexes(cfg, only_news=is_only_news, news_sources_files=news_sources_files)
        save_indexes(domain2ix, only_news=is_only_news, output_path=output_path)
        create_and_save_graph(cfg, domain2ix, graph_type,
                              news_sources_files=news_sources_files,
                              output_path=output_path)
        del domain2ix


if __name__ == '__main__':
    main()
