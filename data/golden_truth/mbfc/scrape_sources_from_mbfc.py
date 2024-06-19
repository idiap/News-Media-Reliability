# -*- coding: utf-8 -*-
"""
Script to scrape the whole mediabiasfactcheck.com site storing media annotation in csv file.

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
import requests
import unicodedata
import pandas as pd

from tqdm import tqdm
from tldextract import extract
from bs4 import BeautifulSoup, SoupStrainer

BASE_URL = "https://mediabiasfactcheck.com"
MBFC_BIAS_CATEGORY = ["center", "left", "leftcenter", "right-center",
                      "right", "conspiracy", "fake-news", "pro-science", "satire"]
OUTPUT_FILE_NAME = "mbfc"  # CSV file name
ERROR_FILE = "error_log.txt"
CORR_COLUMNS = ["factual_reporting", "mbfc_credibility_rating", "nela_gt_label", "bias", "press_freedom", "popularity"]

output_file_name_raw = f"{OUTPUT_FILE_NAME}[raw].csv"
if not os.path.exists(output_file_name_raw):
    error_log = open(ERROR_FILE, "w")
    mbcf_sources = open(output_file_name_raw, "w")
    mbcf_sources.write(f"source,country,bias,factual_reporting,press_freedom,media_type,popularity,mbfc_credibility_rating\n")


    def error(msg):
        error_log.write(f"{msg}\n")
        error_log.flush()


    def get_url_domain(url, full_output=False):
        ext = extract(url) # subdomain, domain, suffix
        output = ext.registered_domain

        if output and full_output and ext.subdomain:
            output = '.'.join(ext)

        return output if ext else None


    for bias in tqdm(MBFC_BIAS_CATEGORY, desc="bias"):
        target_url = f"{BASE_URL}/{bias}/"
        r = requests.get(target_url)
        if r.status_code != 200:
            error(f"{target_url}\t{r.status_code}")
            continue
        if "html" not in r.headers['content-type']:
            error(f"{target_url}\t{r.headers['content-type']}")
            continue

        dom = BeautifulSoup(r.text, "html.parser", parse_only=SoupStrainer("table"))

        table_sources = dom.find("table", {"id": "mbfc-table"})
        if not table_sources:
            table_sources = BeautifulSoup(r.text, "html.parser", parse_only=SoupStrainer("article")).find("div", {"class": "entry-content"})

        links = table_sources.find_all("a")
        if not links:
            error(f"{target_url}\tno links")
            continue

        for link_to_source in tqdm(links, desc="sources"):
            target_url = link_to_source['href']

            try:
                r = requests.get(target_url)
            except requests.exceptions.MissingSchema:
                continue

            if r.status_code != 200:
                error(f"{target_url}\t{r.status_code}")
                continue
            if "html" not in r.headers['content-type']:
                error(f"{target_url}\t{r.headers['content-type']}")
                continue

            dom = BeautifulSoup(r.text, "html.parser", parse_only=SoupStrainer("article"))

            pp = dom.find_all("p")
            if not pp:
                error(f"{target_url}\tno pp")
                continue

            target_p_found = False
            for p in dom.find_all("p"):
                m = re.search(r"Sources?:\s*(.+)", p.text, flags=re.IGNORECASE)

                if not m:
                    m = re.search(r"Sources?:?\s*(http.+)", p.text, flags=re.IGNORECASE)

                if m:
                    source = get_url_domain(m.group(1).strip().lower())

                    m = re.search(r"Country:\s*(.+)", p.text, flags=re.IGNORECASE)
                    country = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"Bias Rating:\s*(.+)", p.text, flags=re.IGNORECASE)
                    bias = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"Factual Reporting:\s*(.+)", p.text, flags=re.IGNORECASE)
                    factual_reporting = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"Press Freedom \w+:\s*(.+)", p.text, flags=re.IGNORECASE)
                    press_freedom = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"Media Type:\s*(.+)", p.text, flags=re.IGNORECASE)
                    media_type = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"Traffic/Popularity:\s*(.+)", p.text, flags=re.IGNORECASE)
                    popularity = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    m = re.search(r"MBFC Credibility Rating:\s*(.+)", p.text, flags=re.IGNORECASE)
                    mbfc_credibility_rating = unicodedata.normalize('NFKC', m.group(1).lower()).strip() if m else ''

                    mbcf_sources.write(f'"{source}","{country}","{bias}","{factual_reporting}","{press_freedom}","{media_type}","{popularity}","{mbfc_credibility_rating}"\n')
                    mbcf_sources.flush()
                    target_p_found = True
                    break

            if not target_p_found:
                error(f"{target_url}\tno target_p_found")

    mbcf_sources.close()
    error_log.close()

else:
    print(f"[!] output file already created ('{output_file_name_raw}'), delete it to scrape values again...")

df = pd.read_csv(output_file_name_raw)
df.fillna('', inplace=True)
df.drop_duplicates(inplace=True)

print(df.head())
print()
# print(df.info())
# print()
print("Total entries:", len(df))
print("Entries with Fact Reporting:", len(df[df.factual_reporting != '']))
print("Entries with MBFC Credibility Rating:", len(df[df.mbfc_credibility_rating != '']))
print()
print("Fact Reporting unique values:", df.factual_reporting.unique().tolist())
print("MBFC Credibility Rating unique values:", df.mbfc_credibility_rating.unique().tolist())

df.factual_reporting = df.factual_reporting.map(lambda v: v.replace('-', ' '))
print("Fixing Fact Reporting unique values:", df.factual_reporting.unique().tolist())
print()
print("Fact Reporting value distribution:")
print(df[df.factual_reporting != ''].factual_reporting.value_counts())
print()
print("MBFC Credibility Rating value distribution:")
print(df[df.mbfc_credibility_rating != ''].mbfc_credibility_rating.value_counts())

# There's only 1 instance with "mixed credibility" for MBFC Credibility Rating.
# I'll replace it's value by 'low credibility'
df.loc[df.mbfc_credibility_rating == 'mixed credibility', 'mbfc_credibility_rating'] = 'low credibility'
print("MBFC Credibility Rating value FINAL distribution:")
print(df[df.mbfc_credibility_rating != ''].mbfc_credibility_rating.value_counts())

# replace non-standard Fact Reportirng `mostly factual` value with `mixed` since they both have the same
# MBFC Credibility Rating distribution
df.loc[df.factual_reporting == 'mostly factual', 'factual_reporting'] = 'mixed'
print()
print("Fact Reporting value FINAL distribution:")
print(df[df.factual_reporting != ''].factual_reporting.value_counts())

# I'll add ground truth label as in NELA-GT-2019.
# 1) From factual reporting column:
#  'unreliable' if factual reporting is "low" or "very low"
#  'mixed' if factual reporting is 'mixed', and
#  'reliable' if the factual reporting is "high" or "very high".
df["nela_gt_label"] = df.factual_reporting.map(lambda v: "unreliable" if "low" in v else ("reliable" if "high" in v else v))
# 2) From bias type: 'unreliable' if flagged as conspiracy or pseudoscience
for ix, row in df.iterrows():
    if 'conspiracy' in row.bias or 'pseudo' in row.bias:
        row.nela_gt_label = "unreliable"

df = df[df.nela_gt_label != '']
df.reset_index(inplace=True)

sources_no_labels = []
for ix, row in df.iterrows():
    if df.iloc[ix, 2:].eq('').all():
        sources_no_labels.append(df.iloc[ix, 0])
df = df[~df.source.isin(sources_no_labels)]

df.to_csv(f"{OUTPUT_FILE_NAME}.csv", index=False)
