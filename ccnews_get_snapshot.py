# -*- coding: utf-8 -*-
"""
Copyright 2022 Idiap Research Institute

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
import hydra

from tqdm import tqdm
from omegaconf import DictConfig
from tools.util import get_url_domain, download_frm_cc, get_sge_job_id_and_max, get_data_split


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg:DictConfig) -> None:
    ccnews_url_domain = get_url_domain(cfg.snapshot.paths_url, full_output=True)
    paths, _ = download_frm_cc(cfg.snapshot.paths_url, decompress=True)

    with open(paths, "r") as reader:
        paths = [l for l in reader.read().split('\n') if l]

    job_id, job_max = get_sge_job_id_and_max()
    if job_id is None:
        job_id = cfg.job.id
        job_max = cfg.job.max

    paths = get_data_split(paths, job_id, job_max)
    for line in tqdm(paths, desc="Downloading WARC files"):
        if not line:
            continue
        download_frm_cc(f"{ccnews_url_domain}/{line}")


if __name__ == '__main__':
    main()
