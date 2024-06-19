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
import os
import json
import hydra
import pysbd
import logging
import fasttext

from tqdm import tqdm
from io import BytesIO
from omegaconf import DictConfig
from typing import Tuple, List
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator
from tools.util import GzippingWrapper, get_url_domain, download_frm_cc, remove_downloaded_frm_cc,\
                                        wait_until_file_is_free, get_data_split,\
                                        get_sge_job_id_and_max, get_wet_writer, \
                                        strip_all


STR_LANG_UNK = "unknown"
MIN_LINE_LEN = 3

Lang_clf = None
Lang_sbd = {}

def get_lang(text:str) -> Tuple[str, float, str]:
    if len(text) > 3:
        try:
            lang_lab, lang_prob = Lang_clf.predict(text.replace('\n', ' '))
        except TypeError:
            text = text.encode("utf-8", "ignore").decode("utf-8")
            lang_lab, lang_prob = Lang_clf.predict(text.replace('\n', ' '))
        lang_lab, lang_prob = lang_lab[0], lang_prob[0]
        if lang_prob > .5:
            return lang_lab.replace("__label__", ""), lang_prob, text
    return STR_LANG_UNK, .0, text


def get_done_file(cache_folder:str, output_folder:str) -> Tuple[str, List[str]]:
    output_folder = os.path.join(cache_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    done_path = os.path.join(output_folder, "done")
    with open(done_path, 'a+') as reader:
        reader.seek(0)
        done_list = reader.read().split('\n')
    return done_path, done_list


def get_passages(text:str, lang:str, max_length:int, sentence_boundaries:bool) -> str:
    if sentence_boundaries and lang not in Lang_sbd:
        try:
            Lang_sbd[lang] = pysbd.Segmenter(language=lang, clean=False)
        except ValueError:
            pass

    if lang in Lang_sbd:
        sentences = Lang_sbd[lang].segment(text)
        sentence_ix = 0
        while sentence_ix < len(sentences):
            passage = ""
            passage_len = 0
            next_sentence_len = len(sentences[sentence_ix].split())
            while passage_len + next_sentence_len <= max_length:
                passage += sentences[sentence_ix]
                passage_len += next_sentence_len

                sentence_ix += 1
                next_sentence_len = len(sentences[sentence_ix].split())\
                                    if sentence_ix < len(sentences)\
                                    else max_length + 1  # forcing loop exiting
            if not passage:
                passage = sentences[sentence_ix]
                sentence_ix += 1
            yield passage
    else:
        tokens = text.split()
        passage_n = len(tokens) // max_length
        for i in range(passage_n):
            yield ' '.join(tokens[i * max_length : (i + 1) * max_length])
        if tokens[passage_n * max_length :]:
            yield ' '.join(tokens[passage_n * max_length :])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg:DictConfig) -> None:
    global Lang_clf

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    ccnews_url_domain = get_url_domain(cfg.snapshot.paths_url, full_output=True)
    os.makedirs(cfg.lang_id.download_path, exist_ok=True)
    model_path = os.path.join(cfg.lang_id.download_path, os.path.split(cfg.lang_id.fasttext_model_url)[1])
    if not os.path.exists(model_path):
        log.info("[downloading] FastText pre-trained language identification model...")
        os.system(f"wget -P {cfg.lang_id.download_path} {cfg.lang_id.fasttext_model_url}")
    wait_until_file_is_free(model_path)
    lang_filter = cfg.extract.lang_filter
    Lang_clf = fasttext.load_model(model_path)

    paths, cache_folder = download_frm_cc(cfg.snapshot.paths_url, decompress=True)

    if cfg.extract.wet:
        wet_done_path, wet_done_list = get_done_file(cache_folder, cfg.output.wet_folder)
        if cfg.clean_cache:
            os.remove(wet_done_path)
            wet_done_list = []
    if cfg.extract.metadata:
        metadata_done_path, metadata_done_list = get_done_file(cache_folder, cfg.output.metadata_folder)
        if cfg.clean_cache:
            os.remove(metadata_done_path)
            metadata_done_list = []
    if cfg.extract.ccnet_json:
        ccnet_done_path, ccnet_done_list = get_done_file(cache_folder, cfg.output.ccnet_folder)
        if cfg.clean_cache:
            os.remove(ccnet_done_path)
            ccnet_done_list = []
    if cfg.extract.index:
        index_done_path, index_done_list = get_done_file(cache_folder, cfg.output.index_folder)
        if cfg.clean_cache:
            os.remove(index_done_path)
            index_done_list = []
    if cfg.extract.news_sources:
        news_sources_done_path, news_sources_done_list = get_done_file(cache_folder, cfg.output.news_sources_folder)
        if cfg.clean_cache:
            os.remove(news_sources_done_path)
            news_sources_done_list = []

    with open(paths, "r") as reader:
        paths = [l for l in reader.read().split('\n') if l]

    job_id, job_max = get_sge_job_id_and_max()
    if job_id is None:
        job_id = cfg.job.id
        job_max = cfg.job.max

    paths = get_data_split(paths, job_id, job_max)
    for url_path in tqdm(paths, desc="Processing WARC files"):
        done = True
        if cfg.extract.wet:
            done = done and url_path in wet_done_list
        if cfg.extract.metadata:
            done = done and url_path in metadata_done_list
        if cfg.extract.ccnet_json:
            done = done and url_path in ccnet_done_list
        if cfg.extract.index:
            done = done and url_path in index_done_list
        if cfg.extract.news_sources:
            done = done and url_path in news_sources_done_list
        if done:
            log.info(f"[skipping] '{url_path}' already processed.")
            continue

        warc_url = f"{ccnews_url_domain}/{url_path}"
        warc_path, _ = download_frm_cc(warc_url, wait_and_retry=True)
        warc_folder, warc_filename_full = os.path.split(warc_path)
        warc_filename = warc_filename_full.split('.')[0]

        if cfg.extract.wet:
            wet_folder = os.path.join(warc_folder, cfg.output.wet_folder)
            wet_filename = os.path.splitext(warc_filename_full)[0] + ".wet.gz"
            os.makedirs(wet_folder, exist_ok=True)
            wet_writer = get_wet_writer(os.path.join(wet_folder, wet_filename))

        output_metadata = {}
        output_ccnet = {}
        output_index = {}
        output_news_sources = {}
        with open(warc_path, 'rb') as stream:
            for record in tqdm(ArchiveIterator(stream), desc="Extracting data", leave=False):
                if record.rec_type == 'response' and record.http_headers.get_header('content-type') and \
                   'html' in record.http_headers.get_header('content-type'):
                    html = record.content_stream().read().decode('utf-8', 'ignore')
                    try:
                        html_dom = BeautifulSoup(html, features="html.parser")
                    except:
                        log.info(f"[skipping] error while trying to parse the HTML content of one the records of the WARC file ({url_path})")
                        continue

                    target_url = record.rec_headers.get_header('WARC-Target-URI')
                    date = record.rec_headers.get_header('WARC-Date')
                    id = record.rec_headers.get_header('WARC-Record-ID')

                    if cfg.extract.index or cfg.extract.ccnet_json or cfg.extract.metadata or cfg.extract.news_sources:
                        html_title = html_dom.find('title')
                        html_title = strip_all(html_title.get_text(strip=True)) if html_title else ''

                    if cfg.extract.wet:
                        html_text = strip_all(html_dom.get_text(separator='\n', strip=True))
                        wet_record = wet_writer.create_warc_record(
                            target_url,
                            'conversion',
                            warc_content_type="text/plain",
                            warc_headers_dict={
                                'WARC-Refers-To': id,
                                'WARC-Date': date
                            },
                            payload=BytesIO(html_text.encode("utf-8", 'ignore')),
                        )
                        # I'll match the exact same headers and order to match the hard-coded WET parser
                        # Facebook people have in the CC-Net source code.
                        # (https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/process_wet_file.py#L57)
                        # Note that I had to also remove the (automatically generated) payload digest
                        # as well (see little hack at the beginning)
                        wet_record.rec_headers.headers = [
                            (h, wet_record.rec_headers.get_header(h))
                            for h in
                            ['WARC-Type', 'WARC-Target-URI',
                             'WARC-Date', 'WARC-Record-ID', 'WARC-Refers-To']
                        ]
                        wet_writer.write_record(wet_record)

                    lang = STR_LANG_UNK
                    if cfg.extract.index or cfg.extract.ccnet_json:
                        digest = record.rec_headers.get_header('WARC-Block-Digest')

                        html_body = html_dom.find('body')
                        doc = strip_all(html_body.get_text(separator='\n', strip=True)) if html_body else ''

                        lang, lang_score, doc = get_lang(doc)

                        if lang_filter and lang not in lang_filter:
                            continue

                        doc = [l for l in doc.split('\n') if len(l) > MIN_LINE_LEN]
                        raw_content = '\n'.join(doc)
                        
                        if cfg.extract.ccnet_json:
                            ccnet_json = {
                                "url": target_url,
                                "date_download": date,
                                "digest": digest,
                                "length": None,
                                "nlines":len(doc),
                                "source_domain": get_url_domain(target_url, full_output=True),
                                "title": html_title,
                                "raw_content": raw_content,
                                "language": lang,
                                "language_score": round(lang_score, 2),
                            #   "perplexity": round(255.11, 1),
                            #   "bucket": "head" # "middle", "tail"
                            }
                            ccnet_json["length"] = len(ccnet_json["raw_content"])

                            if lang not in output_ccnet:
                                ccnet_folder = os.path.join(cache_folder,
                                                            cfg.output.ccnet_folder,
                                                            lang)
                                ccnet_file = "%s.ccnet_%s.jsonl.gz" % (warc_filename, lang)
                                os.makedirs(ccnet_folder, exist_ok=True)
                                output_ccnet[lang] = GzippingWrapper(
                                    open(os.path.join(ccnet_folder, ccnet_file), "wb")
                                )
                            output_ccnet[lang].write(
                                (json.dumps(ccnet_json, separators=(',',':')) + '\n').encode('utf-8')
                            )

                        if cfg.extract.index:
                            if lang not in output_index:
                                index_folder = os.path.join(cache_folder,
                                                            cfg.output.index_folder,
                                                            lang, "index_jsonl")
                                index_file = "%s.index_%s.jsonl.gz" % (warc_filename, lang)
                                os.makedirs(index_folder, exist_ok=True)
                                output_index[lang] = GzippingWrapper(
                                    open(os.path.join(index_folder, index_file), "wb")
                                )

                            index_json = {
                                "id": None,
                                "contents": None,
                                "u": target_url,
                                "d": date,
                                "l": lang,
                                "l_s": round(lang_score, 2),
                                # "p": "perplexity",
                                # "p_b": "bucket"
                            }
                            for ix_p, passage in enumerate(get_passages(raw_content, lang,
                                                                        cfg.index.passage.max_token_length,
                                                                        cfg.index.passage.use_sentence_boundaries)):
                                index_json["id"] = f"{id}{ix_p}"
                                index_json["contents"] = html_title + '[SEP]' + passage

                                output_index[lang].write(
                                    (json.dumps(index_json, separators=(',',':')) + '\n').encode('utf-8', 'ignore')
                                )

                    if cfg.extract.news_sources or cfg.extract.metadata:
                        # Language Identification
                        if lang == STR_LANG_UNK:
                            lang, _, html_title = get_lang(html_title)
                            if lang == STR_LANG_UNK:
                                paragraph_text = ' '.join([p.get_text(separator=' ', strip=True)
                                                           for p in html_dom.find_all('p')])
                                par = strip_all(paragraph_text)
                                lang, _, _ = get_lang(par)

                        if lang_filter and lang not in lang_filter:
                            continue

                        links = []
                        html_body = html_dom.find('body')
                        if html_body:
                            for link in html_body.find_all('a'):
                                if link.has_attr('href'):
                                    links.append(link['href'])

                        if cfg.extract.news_sources:
                            if lang not in output_news_sources:
                                output_news_sources[lang] = defaultdict(lambda: Counter())
                            domain = get_url_domain(target_url)
                            if domain:
                                output_news_sources[lang][domain].update([get_url_domain(l) if get_url_domain(l) else domain for l in links])

                        if cfg.extract.metadata:
                            if lang not in output_metadata:
                                metadata_folder = os.path.join(cache_folder,
                                                               cfg.output.metadata_folder,
                                                               lang)
                                metadata_file = "%s.metadata_%s.jsonl.gz" % (warc_filename, lang)

                                os.makedirs(metadata_folder, exist_ok=True)
                                output_metadata[lang] = GzippingWrapper(
                                    open(os.path.join(metadata_folder, metadata_file), "wb")
                                )

                            metadata_json = {
                                "url": target_url,
                                "title": html_title,
                                "links": links
                            }
                            output_metadata[lang].write((json.dumps(metadata_json, separators=(',',':')) + '\n').encode('utf-8', 'ignore'))

        if cfg.extract.wet:
            wet_writer.close()

        if cfg.extract.news_sources:
            for lang in output_news_sources:
                output_file = os.path.join(cache_folder,
                                           cfg.output.news_sources_folder,
                                           lang)
                os.makedirs(output_file, exist_ok=True)
                output_file = os.path.join(output_file, f"{warc_filename}.news_sources.json.gz")
                output_file = GzippingWrapper(open(output_file, "wb"))
                output_file.write(json.dumps(output_news_sources[lang], separators=(',',':')).encode('utf-8', 'ignore'))
                output_file.close()
            with open(news_sources_done_path, 'a') as writer:
                writer.write(f"{url_path}\n")
                writer.flush()

        if cfg.extract.ccnet_json:
            for lang in output_ccnet:
                output_ccnet[lang].close()
            with open(ccnet_done_path, 'a') as writer:
                writer.write(f"{url_path}\n")
                writer.flush()

        if cfg.extract.index:
            for lang in output_index:
                output_index[lang].close()
            with open(index_done_path, 'a') as writer:
                writer.write(f"{url_path}\n")
                writer.flush()

        if cfg.extract.metadata:
            for lang in output_metadata:
                output_metadata[lang].close()
            with open(metadata_done_path, 'a') as writer:
                writer.write(f"{url_path}\n")
                writer.flush()

        if cfg.extract.remove_originals:
            remove_downloaded_frm_cc(warc_url)

if __name__ == '__main__':
    main()
