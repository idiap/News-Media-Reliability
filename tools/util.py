# -*- coding: utf-8 -*-
"""
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
import zlib
import psutil
import logging
import numpy as np
import networkx as nx

from time import sleep
from hydra import compose
from tldextract import extract
from typing import Optional, Tuple
from warcio.warcwriter import WARCWriter

SLEEP_AND_RETRY_TIME = 60

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GzippingWrapper(object):
    def __init__(self, out):
        self.compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS + 16)
        self.out = out

    def write(self, buff):
        #if isinstance(buff, str):
        #    buff = buff.encode('utf-8')
        buff = self.compressor.compress(buff)
        self.out.write(buff)

    def flush(self, mode=zlib.Z_NO_FLUSH):  # zlib.Z_FULL_FLUSH
        buff = self.compressor.flush(mode)
        self.out.write(buff)
        self.out.flush()

    def close(self):
        self.flush(mode=zlib.Z_FINISH)
        self.out.close()


def investment(graph:nx.DiGraph, rewards:dict,
               max_iter:int=10000, verbose:bool=True) -> np.ndarray:

    if verbose:
        print(f"Investment algorithm begins...")

    V = np.zeros(max(graph.nodes) + 1)
    V_vote = np.zeros_like(V)
    V[list(rewards.keys())] = list(rewards.values())

    for _ in range(max_iter):
        # Investment step
        for id in graph.nodes:
            in_edges = graph.in_edges(id, data=True)
            vote = 0
            for s, _, d in in_edges:
                vote += V[s] * d["weight"]
            V_vote[id] = vote
            # V_vote[id] = vote ** 1.2

        # Truth computation step
        V += V_vote
        for id in graph.nodes:
            out_edges = graph.out_edges(id, data=True)
            # V[id] += V_vote[id]
            for _, t, d in out_edges:
                # V[id] += V_vote[t] * d["weight"]  # H^i(c)
                V[id] += d["weight_in"] * V_vote[t] * d["weight"]  # H^i(c)

    return V


def value_iteration(graph:nx.DiGraph, rewards:dict, only_neg_rewards:bool=False,
                    gamma:float=.8, epsilon:float=1e-6, max_iter:int=10000,
                    verbose:bool=True) -> np.ndarray:

    if verbose:
        print(f"Value iteration begins{' (only negative rewards)' if only_neg_rewards else ''}...")

    V = np.zeros(max(graph.nodes) + 1)
    V_prev = np.zeros_like(V)
    R = np.zeros_like(V)
    R[list(rewards.keys())] = list(rewards.values())

    if only_neg_rewards:
        R[R > 0] = 0  # leaving only negative values

    for iter in range(max_iter):
        for s in graph.nodes:
            v_expected = 0
            out_edges = graph.out_edges(s, data=True)
            for _, t, d in out_edges:
                v_expected += d["weight"] * V_prev[t]
            V[s] = R[s] + gamma * v_expected

        delta = np.abs(V - V_prev).max()
        V_prev[:] = V

        if delta < epsilon:
            if verbose:
                print(f"Value iteration converged after {iter + 1} iterations")
            break

    return V


def reverse_bellman_iteration(graph:nx.DiGraph, rewards:dict, only_pos_rewards:bool=True,
                              use_inbound_prob:bool=False,
                              gamma:float=.8, epsilon:float=1e-6, max_iter:int=10000,
                              verbose:bool=True) -> np.ndarray:

    if verbose:
        print("Reverse Bellman iteration begins (only positive rewards)...")

    V = np.zeros(max(graph.nodes) + 1)
    V_prev = np.zeros_like(V)
    R = np.zeros_like(V)
    R[list(rewards.keys())] = list(rewards.values())
    if only_pos_rewards:
        R[R < 0] = 0  # leaving only positive values

    for iter in range(max_iter):
        for s in graph.nodes:
            vp_expected = 0
            in_edges = graph.in_edges(s, data=True)
            if not use_inbound_prob:
                for p, _, d in in_edges:
                    vp_expected += d["weight"] * V_prev[p]
            else:
                for p, _, d in in_edges:
                    vp_expected += d["weight_in"] * V_prev[p]
            V[s] = R[s] + gamma * vp_expected

        delta = np.abs(V - V_prev).max()
        V_prev[:] = V

        if delta < epsilon:
            if verbose:
                print(f"Reverse Bellman iteration converged after {iter + 1} iterations")
            break

    return V


def get_sge_job_id_and_max() -> Tuple[Optional[int], Optional[int]]:
    """return the normalized jod id and total number of jobs in the job array."""
    try:
        id = int(os.getenv('SGE_TASK_ID'))
        first = int(os.environ.get('SGE_TASK_FIRST'))
        last = int(os.environ.get('SGE_TASK_LAST'))
        step = int(os.environ.get('SGE_TASK_STEPSIZE'))

        job_ids = list(range(first, last + 1, step))
        return job_ids.index(id), len(job_ids)
    except TypeError:
        return None, None


def get_data_split(data:list, id:int=0, max_id:int=1) -> list:
    split_size = round(len(data) / max_id)
    if id >= max_id - 1:
        return data[id * split_size:]
    return data[id * split_size : (id + 1) * split_size]


def wait_until_file_is_free(file_path:str) -> None:
    is_being_used = True
    while is_being_used:
        is_being_used = False
        for proc in psutil.process_iter():
            try:
                for item in proc.open_files():
                    if file_path == item.path:
                        is_being_used = True
                        log.info(f"    [waiting] cached file is still being downloaded, trying again in {SLEEP_AND_RETRY_TIME}s...")
                        sleep(SLEEP_AND_RETRY_TIME)
            except Exception:
                pass


def get_url_domain(url:str, full_output:bool=False) -> Optional[str]:
    ext = extract(url) # subdomain, domain, suffix
    output = ext.registered_domain

    if output and full_output and ext.subdomain:
        output = '.'.join(ext)

    return output if ext else None


def download_frm_cc(url:str, decompress:bool=False, wait_and_retry:bool=False, verbose:bool=True) -> Tuple[str, str]:
    cfg = compose(config_name="config")
    resource_path = url.split(cfg.snapshot.url_split)[1]

    if decompress:
        resource_path, resource_ext = os.path.splitext(resource_path)

    cache_directory = os.path.split(resource_path)[0]

    if verbose:
        log.info(f"[downloading] {resource_path}...")

    os.makedirs(cache_directory, exist_ok=True)
    if not os.path.exists(resource_path):
        # try -p {url} to see if it creates the path locally
        os.system(f"wget -P {cache_directory} {url}")

        if decompress:
            os.system(f"gzip -d {resource_path}{resource_ext}")
    else:
        log.info("    [skipped] already cached")
        if wait_and_retry:
            wait_until_file_is_free(resource_path)

    return resource_path, cache_directory


def remove_downloaded_frm_cc(url:str):
    cfg = compose(config_name="config")
    os.system(f"rm {os.path.splitext(url.split(cfg.snapshot.url_split)[1])[0]}*")


def strip_all(text:str) -> str:
    return re.sub(r"(\s)\s*", r"\g<1>", text).strip()


def get_wet_writer(wet_path:str) -> WARCWriter:

    # Little hack to prevent payload digest header from being automatically generated
    def force_disable_payload_digest(ensure_digest):
        def call(*argv, **kw):
            kw["payload"] = False
            return ensure_digest(*argv, **kw)
        return call

    # Adding a "close" method to close the wrapped file -> this should be already implemented :/
    if not hasattr(WARCWriter, 'close'):
        WARCWriter.close = lambda self: self.out.close()

    # wet_file = open(wet_path, 'wb')
    wet_writer = WARCWriter(open(wet_path, 'wb'), gzip=True)
    wet_writer.ensure_digest = force_disable_payload_digest(wet_writer.ensure_digest)

    return wet_writer
