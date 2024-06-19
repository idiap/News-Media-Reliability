# -*- coding: utf-8 -*-
"""
Script to generate the dataset containing the golden truth values

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
import json
import hydra
import logging
import requests
import pandas as pd

from bs4 import BeautifulSoup, SoupStrainer
from collections import defaultdict
from omegaconf import DictConfig
from tools.util import get_url_domain


STATUS2RELIABILITY = {
    "generally reliable": 1,
    "no consensus": 0,
    "generally unreliable": -1,
    "deprecated": -1,
    "blacklisted": -2,
}
NELALABEL2RELIABILITY = {
    "reliable": 1,
    "mixed": 0,
    "unreliable": -1
}


def status2reliability(status_column_links):
    return min([STATUS2RELIABILITY[a['title'].lower()] for a in status_column_links])


def uses2domains(uses_column_links):
    domains = []
    for link in uses_column_links:
        if not link.text:
            continue
        try:
            int(link.text)
            if link.text.startswith('+'):
                continue
            domains.append(get_url_domain(re.search(r'"(.+)"', link["title"]).group(1)))
        except ValueError:
            pass
    return domains


def get_next_line(dreader, use_gz=True):
    line = True
    while line:
        line = dreader.readline().decode('utf-8') if use_gz else dreader.readline()
        if line:
            yield line


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg:DictConfig) -> None:
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger(__name__)

    cfg_auto = cfg.golden_truth.automatic
    cfg_wiki = cfg_auto.wikipedia
    opath = cfg.golden_truth.output_path
    os.makedirs(opath, exist_ok=True)

    # 1. Perennial sources from Wikipedia
    if cfg_auto.rebuild:
        print("Extracting perennial sources from Wikipedia...")
        r = requests.get(cfg_wiki.perennial_sources.url)
        html_dom = BeautifulSoup(r.text, parse_only=SoupStrainer("div", attrs={"class":"mw-parser-output"}), features="html.parser")

        rows = [tr.find_all("td") for tr in html_dom.find("table", attrs={"class":"sortable"}).find("tbody").find_all("tr")]
        rows = [(uses2domains(r[5].find_all("a")), status2reliability(r[1].find_all("a"))) for r in rows if r]

        perennial_sources = {}
        for domains, reliability_degree in rows:
            for domain in domains:
                if domain and domain not in perennial_sources:
                    perennial_sources[domain] = reliability_degree
        with open(os.path.join(opath, cfg_wiki.perennial_sources.output_file), 'w') as writer:
            json.dump(perennial_sources, writer)
    else:
        with open(os.path.join(opath, cfg_wiki.perennial_sources.output_file)) as reader:
            perennial_sources = json.load(reader)
    print("# Perennial sources:", len(perennial_sources))

    # 2. NewsGuard values from NELA-GT-2018
    if cfg_auto.rebuild:
        df = pd.read_csv(os.path.join(opath, cfg_auto.nela_gt.input_file))
        df = df[~df["NewsGuard, overall_class"].isna()]
        df.rename(columns={"NewsGuard, overall_class": "reliability_label", "NewsGuard, score": "reliability_score"}, inplace=True)
        df_newsguard = df[["domain", "reliability_label", "reliability_score"]]
        df_newsguard.to_csv(
            os.path.join(opath, cfg_auto.nela_gt.output_newsguard_file),
            index=False
        )
    else:
        with open(os.path.join(opath, cfg_auto.nela_gt.output_newsguard_file)) as reader:
            df_newsguard = pd.read_csv(reader)
    newsguard_sources = {}
    newsguard_scores = defaultdict(lambda:None)
    for _, row in df_newsguard.iterrows():
        newsguard_sources[row.domain] = STATUS2RELIABILITY["generally reliable"] if row.reliability_label else STATUS2RELIABILITY["generally unreliable"]
        newsguard_scores[row.domain] = row.reliability_score
    print("# NewsGuard sources:", len(newsguard_scores))


    # 3. Media Bias/Fact Check (MBFC) sources
    with open(os.path.join(opath, cfg_auto.mbfc.path)) as reader:
        df_mbfc = pd.read_csv(reader)
    mbfc_sources = {}  # use df_newsguard to populate
    for _, row in df_mbfc.iterrows():
        mbfc_sources[row.source] = NELALABEL2RELIABILITY[row.nela_gt_label]
    print("# MBFC sources:", len(mbfc_sources))

    # 4. User-provided reliable sources list (currently empty)
    manual_reliable_sources = {}
    if os.path.exists(cfg.golden_truth.manual.reliable_sources_path):
        print("Loading list of reliable news sources...")
        with open(cfg.golden_truth.manual.reliable_sources_path) as reader:
            for line in reader.read().split('\n'):
                domain = get_url_domain(line.strip())
                if not domain:
                    continue
                manual_reliable_sources[domain] = STATUS2RELIABILITY["generally reliable"]
    print("# User-provided reliable sources:", len(manual_reliable_sources))

    # 5. User-provided unreliable sources list (collected from different papers and documents)
    manual_unreliable_sources = {}
    if os.path.exists(cfg.golden_truth.manual.unreliable_sources_path):
        print("Loading list of unreliable news sources...")
        with open(cfg.golden_truth.manual.unreliable_sources_path) as reader:
            for line in reader.read().split('\n'):
                domain = get_url_domain(line.strip())
                if not domain:
                    continue

                manual_unreliable_sources[domain] = STATUS2RELIABILITY["generally unreliable"]

    print("# User-provided unreliable sources:", len(manual_unreliable_sources))

    golden_truth_file = os.path.join(opath, cfg.golden_truth.output_file)
    print(f"Creating files with golden truth reliability values ('{golden_truth_file}')...")
    # Creating final dataset. Label priorities from lowest to highest:
    # 1. NewsGuard values from NELA-GT-2018
    # 2. User-provided reliable sources list (currently empty)
    # 3. User-provided unreliable sources list (collected from different papers and documents)
    # 4. Perennial sources from Wikipedia
    # 5. Media Bias/Fact Check (MBFC) sources
    news_sources_reliability_info = mbfc_sources
    news_sources_reliability_info.update(perennial_sources)
    news_sources_reliability_info.update(manual_unreliable_sources)
    news_sources_reliability_info.update(manual_reliable_sources)
    news_sources_reliability_info.update(newsguard_sources)

    label_sources = defaultdict(lambda:None)
    for s in mbfc_sources:
        label_sources[s] = "mbfc"
    for s in perennial_sources:
        label_sources[s] = "wikipedia"
    for s in manual_unreliable_sources:
        label_sources[s] = "manual"
    for s in manual_reliable_sources:
        label_sources[s] = "manual"
    for s in newsguard_sources:
        label_sources[s] = "newsguard"

    # Adding news citations from Wikipedia
    # Adding NewsGuard scores
    rows_df = []
    for domain, label in news_sources_reliability_info.items():
        rows_df.append({"domain": domain,
                        "reliability_label": label,
                        "newsguard_score": newsguard_scores[domain],
                        "label_source": label_sources[domain]})
    df_dataset = pd.DataFrame.from_dict(rows_df)
    df_dataset.to_csv(golden_truth_file, index=False)

    print("\nLabel distribution:")
    print(df_dataset.reliability_label.value_counts(sort=False))

    print("\nLabel source distribution:")
    print(df_dataset.label_source.value_counts(sort=False))

if __name__ == '__main__':
    main()
