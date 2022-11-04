#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:14:47 2022

@author: daniel, arthur
"""

from multiprocessing.spawn import prepare
import threading
from alive_progress import alive_bar
from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import shutil
import regex
import os
import re
import itertools as it
import more_itertools as mit
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))


class Data:
    def __init__(self):

        self.available_databases = [
            "Opinosis",
            "CNN_DailyMail",
            "Big_Patent",
            "XSum",
            "CNN_Corpus_Extractive",
            "CNN_Corpus_Abstractive",
            "ArXiv_PubMed",
            "ArXiv",
            "PubMed",
        ]

    def show_available_databases(self):
        print()
        print("The available databases are:")
        for i, database in enumerate(self.available_databases):
            print(str(i) + ": " + database)
        print()

    def wipe_cache(self):
        data = os.listdir(os.path.join(ROOT_DIR, "data"))
        for item in data:
            if item.endswith(".lock"):
                os.remove(os.path.join(ROOT_DIR, "data", item))
            if item == "downloads":
                shutil.rmtree(os.path.join(ROOT_DIR, "data", item))

    def _clean_text(self, content):
        if isinstance(content, str):
            pass
        else:
            content = str(content)
        # strange jump lines
        content = re.sub(r"\.", ". ", str(content))
        # trouble characters
        content = re.sub(r"\\r\\n", " ", str(content))
        # clean jump lines
        content = re.sub(r"\u000D\u000A|[\u000A\u000B\u000C\u000D\u0085\u2028\u2029]", " ", content)
        # Replace different spaces
        content = re.sub(r"\u00A0\u1680​\u180e\u2000-\u2009\u200a​\u200b​\u202f\u205f​\u3000", " ", content)
        # replace multiple spaces
        content = re.sub(r" +", " ", content)
        # normalize hiphens
        content = regex.sub(r"\p{Pd}+", "-", content)
        # normalize single quotations
        content = re.sub(r"[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]", "'", content)
        # normalize double quotations
        content = re.sub(r"[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]", '"', content)
        # normalize apostrophes
        content = re.sub(r"[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "'", content)
        return content

    def read_data(self, database_name, size=10000):
        self.database_name = database_name
        self.size = size
        self.the_data_reader = getattr(self, "read_" + database_name.lower())
        self.the_data_reader()
        self.wipe_cache()
        return self.data

    ### Huggingface Datasets ###

    def read_opinosis(self):
        dataset = load_dataset("opinosis", split="train", cache_dir=os.path.join(ROOT_DIR, "data"))
        summaries_dict = {}
        summaries_df = pd.DataFrame(columns=[])
        texts = []
        with alive_bar(len(dataset), title="Loading Opinosis...") as bar:
            for i in range(0, len(dataset)):
                texts.append(" ".join(dataset[i]["review_sents"].split()))
                for j in range(0, len(dataset[i]["summaries"])):
                    summaries_dict["summary " + str(j)] = " ".join(dataset[i]["summaries"][j].split())
                summaries_df = pd.concat([summaries_df, pd.DataFrame.from_dict([summaries_dict])], ignore_index=True)
                bar()
            texts_df = pd.DataFrame(texts, columns=["text"])
            self.data = pd.concat([texts_df, summaries_df], axis=1)
            self.data = self.data.head(self.size)
            # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    def read_cnn_dailymail(self):
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        texts, summaries = [], []
        with alive_bar(len(dataset), title="Loading CNN/Daily Mail...") as bar:
            for i in range(0, len(dataset)):
                texts.append(" ".join(dataset[i]["article"].split()))
                summaries.append(" ".join(dataset[i]["highlights"].split()))
                bar()
            self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
            self.data = self.data.head(self.size)
            # self.data.to_csv(os.path.join(ROOT_DIR,"dataset_test.csv"), index=False)

    def read_big_patent(self):
        with alive_bar(bar=None, monitor=False, stats=False, spinner="dots", title="Loading BIGPATENT..."):
            dataset = load_dataset("big_patent", "all", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
            self.data = dataset.to_pandas()
            self.data = self.data.rename(columns={"description": "text", "abstract": "golden"})
            self.data["text"] = self.data["text"].apply(lambda x: " ".join(self._clean_text(x).split()))
            self.data["golden"] = self.data["golden"].apply(lambda x: " ".join(self._clean_text(x).split()))
            self.data = self.data.head(self.size)
            # self.data.to_csv(os.path.join(ROOT_DIR,"dataset_test.csv"), index=False)

    def read_xsum(self):
        dataset = load_dataset("xsum", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        texts, summaries = [], []
        with alive_bar(len(dataset), title="Loading XSum...") as bar:
            for i in range(0, len(dataset)):
                texts.append(" ".join(dataset[i]["document"].split()))
                summaries.append(" ".join(dataset[i]["summary"].split()))
                bar()
        self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
        self.data = self.data.head(self.size)
        # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    def read_arxiv_pubmed(self):
        pubmed = load_dataset("ccdv/pubmed-summarization", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        arxiv = load_dataset("ccdv/arxiv-summarization", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        texts, summaries = [], []
        with alive_bar(len(pubmed) + len(arxiv), title="Loading PubMed + ArXiv...") as bar:
            for i in range(0, len(pubmed)):
                texts.append(" ".join(pubmed[i]["article"].split()))
                summaries.append(" ".join(pubmed[i]["abstract"].split()))
                bar()
            for i in range(0, len(arxiv)):
                texts.append(" ".join(arxiv[i]["article"].split()))
                summaries.append(" ".join(arxiv[i]["abstract"].split()))
                bar()
        self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
        self.data = self.data.head(self.size)
        # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    def read_arxiv(self):
        arxiv = load_dataset("ccdv/arxiv-summarization", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        texts, summaries = [], []
        with alive_bar(len(arxiv), title="Loading ArXiv...") as bar:
            for i in range(0, len(arxiv)):
                texts.append(" ".join(arxiv[i]["article"].split()))
                summaries.append(" ".join(arxiv[i]["abstract"].split()))
                bar()
        self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
        self.data = self.data.head(self.size)
        # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    def read_pubmed(self):
        pubmed = load_dataset("ccdv/pubmed-summarization", split="test", cache_dir=os.path.join(ROOT_DIR, "data"))
        texts, summaries = [], []
        with alive_bar(len(pubmed), title="Loading PubMed...") as bar:
            for i in range(0, len(pubmed)):
                texts.append(" ".join(pubmed[i]["article"].split()))
                summaries.append(" ".join(pubmed[i]["abstract"].split()))
                bar()
        self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
        self.data = self.data.head(self.size)
        # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    ### Other Datasets ###

    def read_cnn_corpus_abstractive(self):
        texts_df = pd.DataFrame(columns=["text"])
        summaries_df = pd.DataFrame()
        text_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "CNN_Corpus")
        processed_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "processed_abstractive")
        texts = os.listdir(text_path)
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
            with alive_bar(len(texts), title="Loading CNN Corpus (Abstractive)") as bar:
                for file in texts:
                    with open(os.path.join(text_path, file), "r", encoding="utf8") as f:
                        soup = BeautifulSoup(f.read(), "xml")
                    cleanup = {"&quot;": '"', "&apost;": "'"}
                    abs_sum = soup.find("highlights").get_text()
                    text = soup.find("article").get_text()
                    for key, value in cleanup.items():
                        abs_sum = " ".join(abs_sum.replace(key, value).split())
                        text = " ".join(text.replace(key, value).split())
                    bar()
                    texts_df = pd.concat([texts_df, pd.DataFrame({"text": [self._clean_text(text)]})], ignore_index=True)
                    summaries_df = pd.concat([summaries_df, pd.DataFrame({"golden": [self._clean_text(abs_sum)]})], ignore_index=True)
                    self.data = pd.concat([texts_df, summaries_df], axis=1)
                    self.data.to_csv(os.path.join(processed_path, "CNN_Corpus_Abstractive.csv"), index=False)
                    self.data = self.data.head(self.size)
        else:
            self.data = pd.read_csv(os.path.join(processed_path, "CNN_Corpus_Abstractive.csv"))
            self.data = self.data.head(self.size)

    def read_cnn_corpus_extractive(self):
        texts_df = pd.DataFrame(columns=["text"])
        summaries_df = pd.DataFrame()
        text_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "CNN_Corpus")
        processed_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "processed_extractive")
        texts = os.listdir(text_path)
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
            with alive_bar(len(texts), title="Loading CNN Corpus (Extractive)") as bar:
                for file in texts:
                    with open(os.path.join(text_path, file), "r", encoding="utf8") as f:
                        soup = BeautifulSoup(f.read(), "xml")
                    cleanup = {"&quot;": '"', "&apost;": "'"}
                    ext_sum = soup.find("gold_standard").get_text()
                    text = soup.find("article").get_text()
                    for key, value in cleanup.items():
                        ext_sum = " ".join(ext_sum.replace(key, value).split())
                        text = " ".join(text.replace(key, value).split())
                    bar()
                    texts_df = pd.concat([texts_df, pd.DataFrame({"text": [self._clean_text(text)]})], ignore_index=True)
                    summaries_df = pd.concat([summaries_df, pd.DataFrame({"golden": [self._clean_text(ext_sum)]})], ignore_index=True)
                    self.data = pd.concat([texts_df, summaries_df], axis=1)
                    self.data.to_csv(os.path.join(processed_path, "CNN_Corpus_Abstractive.csv"), index=False)
                    self.data = self.data.head(self.size)
        else:
            self.data = pd.read_csv(os.path.join(processed_path, "CNN_Corpus_Abstractive.csv"))
            self.data = self.data.head(self.size)


class Method:
    def __init__(self, data_df, data_name):
        self.data_name = data_name
        self.texts = data_df["text"].tolist()
        self.golden_summaries = data_df["golden"].tolist()
        self.results = pd.DataFrame(columns=["method", "summary", "golden", "source"])
        self.available_methods = [
            "SumyRandom",
            "SumyLuhn",
            "SumyLsa",
            "SumyLexRank",
            "SumyTextRank",
            "SumySumBasic",
            "SumyKL",
            "SumyReduction",
            "Transformers-google/pegasus-xsum",
            "Transformers-facebook/bart-large-cnn",
            "Transformers-csebuetnlp/mT5_multilingual_XLSum",
        ]
        self.target_lengths = {
            "opinosis": [1, 30],
            "cnn_dailymail": [2, 40],
            "big_patent": [4, 130],
            "cnn_corpus_abstractive": [3, 55],
            "cnn_corpus_extractive": [4, 150],
            "xsum": [1, 20],
            "arxiv_pubmed": [6, 240],
            "arxiv": [6, 240],
            "pubmed": [6, 240],
        }
        self.sentence_count = self.target_lengths.get(self.data_name)[0]
        self.token_count = self.target_lengths.get(self.data_name)[1]

    def show_methods(self):
        print()
        print("The avaliable methods are:")
        for i, method in enumerate(self.available_methods):
            print(str(i) + ": " + method)
        print()

    def examples_to_csv(self, n=10000):
        if not os.path.exists(os.path.join(ROOT_DIR, "results")):
            os.makedirs(os.path.join(ROOT_DIR, "results"))
            self.results.head(n).to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_examples.csv"), index=False)
        elif not os.path.exists(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_examples.csv")):
            self.results.head(n).to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_examples.csv"), index=False)
        else:
            old = pd.read_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_examples.csv"))
            new = pd.concat([old, self.results.head(n)]).drop_duplicates()
            new.to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_examples.csv"), index=False)

    def run(self, the_method):
        self.the_method = the_method
        if self.the_method[0:4] == "Sumy":
            self.run_sumy()
        elif self.the_method[0:13] == "Transformers-":
            self.run_transformers()
        else:
            print("This method is not defined! Try another one.")

        print(f"{len(self.candidate_summaries)} Summaries generated. \n")
        return self.results

    def run_sumy(self):
        from sumy.summarizers.random import RandomSummarizer
        from sumy.summarizers.luhn import LuhnSummarizer
        from sumy.summarizers.lsa import LsaSummarizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        from sumy.summarizers.sum_basic import SumBasicSummarizer
        from sumy.summarizers.kl import KLSummarizer
        from sumy.summarizers.reduction import ReductionSummarizer
        from sumy.summarizers.edmundson import EdmundsonSummarizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer  # For Strings
        from sumy.parsers.html import HtmlParser
        from sumy.utils import get_stop_words
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from collections import Counter

        def word_frequency(summaries, texts, n=20):
            lemmatizer = WordNetLemmatizer()

            summaries = " ".join(summaries)
            sum_tokens = [t.lower() for t in word_tokenize(summaries)]
            sum_tokens = [t for t in sum_tokens if t not in stopwords.words("english")]
            sum_tokens = [t for t in sum_tokens if t.isalpha()]
            # sum_tokens = [lemmatizer.lemmatize(t) for t in sum_tokens]
            sum_counted = Counter(sum_tokens)
            sum_word_freq_descending = pd.DataFrame(sum_counted.items(), columns=["word", "frequency sum"]).sort_values(by="frequency sum", ascending=False)

            texts = " ".join(texts)
            texts_tokens = [t.lower() for t in word_tokenize(texts)]
            texts_tokens = [t for t in texts_tokens if t not in stopwords.words("english")]
            texts_tokens = [t for t in texts_tokens if t.isalpha()]
            # texts_tokens = [lemmatizer.lemmatize(t) for t in texts_tokens]
            texts_counted = Counter(texts_tokens)
            texts_word_freq_descending = pd.DataFrame(texts_counted.items(), columns=["word", "frequency text"]).sort_values(by="frequency text", ascending=False)

            # DF_doc/DF_sum.
            stigma_words = pd.merge(sum_word_freq_descending, texts_word_freq_descending, on="word")
            stigma_words["frequency"] = stigma_words["frequency text"] / stigma_words["frequency sum"]
            stigma_words = stigma_words.sort_values(by="frequency", ascending=False)

            stigma_words = stigma_words["word"].tolist()[:n]
            bonus_words = sum_word_freq_descending["word"].tolist()[:n]
            return bonus_words, stigma_words

        the_method = self.the_method.replace("Sumy", "")
        the_summarizer = locals()[the_method + "Summarizer"]()

        with alive_bar(len(self.texts), bar=None, spinner="dots", title="Running " + self.the_method + " Summarizer") as bar:
            summarizer_output_list = []
            if the_method == "Edmundson":
                bonus_words, stigma_words = word_frequency(self.golden_summaries[:200], self.texts[:200], 10)
            for text in self.texts:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                if the_method != "Edmundson":
                    summarizer_output_list.append(the_summarizer(parser.document, self.sentence_count))
                else:
                    the_summarizer = EdmundsonSummarizer(cue_weight=1, key_weight=1, title_weight=0, location_weight=0)
                    the_summarizer.bonus_words = bonus_words
                    the_summarizer.stigma_words = stigma_words
                    the_summarizer.null_words = stopwords.words("english")
                    summarizer_output_list.append(the_summarizer(parser.document, self.sentence_count))
                bar()

        self.candidate_summaries = []
        for summarizer_output in summarizer_output_list:
            text_summary = ""
            for sentence in summarizer_output:
                text_summary += str(sentence) + " "

            self.candidate_summaries.append(text_summary)

        self.results = pd.DataFrame({"summary": self.candidate_summaries, "golden": self.golden_summaries, "source": self.texts})
        self.results.insert(0, "method", self.the_method)

    def run_transformers(self):
        from transformers import pipeline
        from nltk.tokenize import word_tokenize

        the_method = self.the_method.replace("Transformers-", "")
        with alive_bar(len(self.texts), bar=None, spinner="dots", title="Running Transformers-" + the_method) as bar:
            summarizer = pipeline("summarization", model=the_method)

            self.aux_summaries_list = []
            for text in self.texts:
                length = 3000
                while len(word_tokenize(text[0:length])) > 450:
                    length -= 100
                self.aux_summaries_list.append(summarizer(text[0:length], min_length=(self.token_count - 5), max_length=(self.token_count + 5)))
                bar()

        self.candidate_summaries = [x[0]["summary_text"] for x in self.aux_summaries_list]

        self.results = pd.DataFrame({"summary": self.candidate_summaries, "golden": self.golden_summaries, "source": self.texts})
        self.results.insert(0, "method", self.the_method)

    def import_summaries(self, transformer, dataset):
        IMPORT_DIR = os.path.join(ROOT_DIR, "results", "import")
        if os.path.exists(IMPORT_DIR):
            data = pd.read_csv(os.path.join(IMPORT_DIR, dataset + "_examples.csv"))
            data = data.sort_values(by=["method"], ascending=True).reset_index(drop=True).drop("source", axis=1)
            idx1 = data.method.eq("Transformers-facebook/bart-large-cnn").idxmax()
            idx2 = data.method.eq("Transformers-google/pegasus-xsum").idxmax()
            data_t5, data_bart, data_pegasus = data.iloc[:idx1].drop("method", axis=1), data.iloc[idx1:idx2].drop("method", axis=1), data.iloc[idx2:].drop("method", axis=1)
            data_t5.head()

            print(f"\n{dataset}: Imported {transformer} summaries.\n")
            if transformer == "Transformers-facebook/bart-large-cnn":
                return data_bart
            elif transformer == "Transformers-google/pegasus-xsum":
                return data_pegasus
                pass
            else:
                return data_t5
        else:
            print("No import folder found.")


class Evaluator:
    def __init__(self, data_df, method, data_name):
        self.golden_summaries = data_df["golden"].tolist()
        self.candidate_summaries = data_df["summary"].to_list()
        self.available_evaluators = ["rouge", "nltk", "gensim", "sklearn"]
        self.method = method
        self.data_name = data_name
        self.results_df = pd.DataFrame(columns=["data", "method", "aggregator", "metric", "P", "R", "F1", "H", "J", "KLD", "C"])

    def show_evaluators(self):
        print("The avaliable evaluators are:")
        for i, evaluator in enumerate(self.available_evaluators):
            print(str(i) + ": " + evaluator)
        print()

    def run(self, the_evaluator):
        self.the_evaluator = the_evaluator
        if self.the_evaluator == "rouge":
            self.run_rouge_eval()
        elif self.the_evaluator == "nltk":
            self.run_nltk_eval()
        elif self.the_evaluator == "gensim":
            self.run_gensim_eval()
        elif self.the_evaluator == "sklearn":
            self.run_sklearn_eval()
        else:
            print("This evaluator is not defined! Try another one.")

    def metrics_to_csv(self):
        if not os.path.exists(os.path.join(ROOT_DIR, "results")):
            os.makedirs(os.path.join(ROOT_DIR, "results"))
            self.results_df.to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_results.csv"), index=False)
        elif not os.path.exists(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_results.csv")):
            self.results_df.to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_results.csv"), index=False)
        else:
            old = pd.read_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_results.csv"))
            new = pd.concat([old, self.results_df]).drop_duplicates()
            new.to_csv(os.path.join(ROOT_DIR, "results", str(self.data_name) + "_results.csv"), index=False)

    def join_all_results(self):
        join_df = pd.DataFrame(columns=["data", "method", "aggregator", "metric", "P", "R", "F1", "H", "J", "KLD", "C"])
        results = os.listdir(os.path.join(ROOT_DIR, "results"))
        for file in results:
            if file.endswith("_results.csv") and not file.endswith("all_results.csv"):
                data = pd.read_csv(os.path.join(ROOT_DIR, "results", file))
                join_df = pd.concat([join_df, data]).drop_duplicates()
        join_df.to_csv(os.path.join(ROOT_DIR, "results", "all_results.csv"), index=False)

    def run_rouge_eval(self):
        import rouge
        from rouge_metric import PyRouge

        def prepare_rouge():
            self.references = []
            self.hypotheses = self.candidate_summaries
            for i in range(0, len(self.golden_summaries)):
                self.references.append([self.golden_summaries[i]])

        def prepare_results_for_csv(data, method, aggregator, metric, p, r, f):
            return str(data), str(method), str(aggregator), str(metric), "{:5.2f}".format(100.0 * p), "{:5.2f}".format(100.0 * r), "{:5.2f}".format(100.0 * f)

        def results_concat(aggregator, metric, results, results_df):
            data, method, a, m, p, r, f = prepare_results_for_csv(self.data_name, self.method, aggregator, metric, results["p"], results["r"], results["f"])
            return pd.concat([results_df, pd.DataFrame({"data": [data], "method": [method], "aggregator": [a], "metric": [m], "P": [p], "R": [r], "F1": [f]})], ignore_index=True)

        def print_results(m, p, r, f):
            return "\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}".format(m, "P", 100.0 * p, "R", 100.0 * r, "F1", 100.0 * f)

        prepare_rouge()

        for aggregator in ["Avg"]:
            apply_avg = aggregator == "Avg"
            apply_best = aggregator == "Best"

            evaluator = rouge.Rouge(
                metrics=["rouge-n", "rouge-l", "rouge-w"],
                max_n=4,
                limit_length=True,
                length_limit=100,
                length_limit_type="words",
                apply_avg=apply_avg,
                apply_best=apply_best,
                alpha=0.5,
                weight_factor=1.2,
                stemming=True,
            )  # Default F1_score

            evaluator_su = PyRouge(
                # rouge_n=(1, 2, 3, 4),
                # rouge_l=True,
                # rouge_w=True,
                # rouge_w_weight=1.2,
                # rouge_s=True,
                rouge_su=True,
                skip_gap=4,
            )

            with alive_bar(bar=False, monitor=False, stats=False, spinner="dots", title="Evaluation with ROUGE...") as bar:
                scores = evaluator.get_scores(self.hypotheses, self.references)
                if apply_avg:
                    su = evaluator_su.evaluate(self.hypotheses, self.references)
                    scores = dict(scores, **su)

            print("\t{}:".format(aggregator))

            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                    for hypothesis_id, results_per_ref in enumerate(results):
                        nb_references = len(results_per_ref["p"])
                        for reference_id in range(nb_references):
                            print("\tHypothesis #{} & Reference #{}: ".format(hypothesis_id, reference_id))
                            print("\t" + print_results(metric, results_per_ref["p"][reference_id], results_per_ref["r"][reference_id], results_per_ref["f"][reference_id]))
                    print()
                else:
                    print(print_results(metric, results["p"], results["r"], results["f"]))
                    self.results_df = results_concat(aggregator, metric, results, self.results_df)
            print()

    def run_nltk_eval(self):
        from nltk.metrics.scores import precision, recall, f_measure

        def prepare_nltk():
            self.references, self.hypotheses = [], []
            for i in range(0, len(self.golden_summaries)):
                hypothesis_split = []
                for word in self.candidate_summaries[i].split():
                    hypothesis_split.append(word)
                self.hypotheses.append(hypothesis_split)
                reference_split = []
                for word in self.golden_summaries[i].split():
                    reference_split.append(word)
                self.references.append(reference_split)

        def prepare_results_for_csv(data, method, aggregator, metric, p, r, f):
            return str(data), str(method), str(aggregator), str(metric), "{:5.2f}".format(100.0 * p), "{:5.2f}".format(100.0 * r), "{:5.2f}".format(100.0 * f)

        def results_concat(aggregator, metric, precision, recall, fmeasure, results_df):
            data, method, a, m, p, r, f = prepare_results_for_csv(self.data_name, self.method, aggregator, metric, precision, recall, fmeasure)
            return pd.concat([results_df, pd.DataFrame({"data": [data], "method": [method], "aggregator": [a], "metric": [m], "P": [p], "R": [r], "F1": [f]})], ignore_index=True)

        def print_results(p, r, f, p_m, r_m, f_m):
            print(f"\tAvg:\t\tP: {100*p:5.2f} \tR: {100*r:5.2f} \tF1: {100*f:5.2f}\n\tBest:\t\tP: {100*p_m:5.2f} \tR: {100*r_m:5.2f} \tF1: {100*f_m:5.2f}")
            print()

        p, r, f = [], [], []
        with alive_bar(len(self.candidate_summaries), title="Evaluation with NLTK...", bar=False, spinner="dots") as bar:
            prepare_nltk()
            for i in range(0, len(self.hypotheses)):
                p.append(precision(set(self.references[i]), set(self.hypotheses[i])))
                r.append(recall(set(self.references[i]), set(self.hypotheses[i])))
                f.append(f_measure(set(self.references[i]), set(self.hypotheses[i]), alpha=0.5))
                bar()

            p_avg = sum(p) / len(p)
            r_avg = sum(r) / len(r)
            f_avg = sum(f) / len(f)
            p_best = max(p)
            r_best = max(r)
            f_best = max(f)

        print_results(p_avg, r_avg, f_avg, p_best, r_best, f_best)
        self.results_df = results_concat("Avg", "NLTK", p_avg, r_avg, f_avg, self.results_df)

    def run_gensim_eval(self):
        from gensim.matutils import kullback_leibler, hellinger, jaccard, jensen_shannon
        from gensim.corpora import Dictionary, HashDictionary
        from gensim.models import ldamodel, NormModel

        self.gensim_thread_count = 10000

        def prepare_gensim():
            self.references, self.hypotheses = [], []
            for i in range(0, len(self.golden_summaries)):
                hypothesis_split = []
                for word in self.candidate_summaries[i].split():
                    hypothesis_split.append(word)
                self.hypotheses.append(hypothesis_split)
                reference_split = []
                for word in self.golden_summaries[i].split():
                    reference_split.append(word)
                self.references.append(reference_split)

            self.hypotheses = [list(x) for x in mit.divide(self.gensim_thread_count, self.hypotheses)]
            self.references = [list(x) for x in mit.divide(self.gensim_thread_count, self.references)]

        def prepare_results_for_csv(data, method, aggregator, metric, hellinger_div, jaccard_div, kl_div):
            return str(data), str(method), str(aggregator), str(metric), "{:5.2f}".format(jaccard_div), "{:5.2f}".format(hellinger_div), "{:5.2f}".format(kl_div)

        def results_concat(aggregator, metric, hellinger_div, jaccard_div, kl_div, results_df):
            data, method, a, m, j, h, kld = prepare_results_for_csv(self.data_name, self.method, aggregator, metric, hellinger_div, jaccard_div, kl_div)
            return pd.concat([results_df, pd.DataFrame({"data": [data], "method": [method], "aggregator": [a], "metric": [m], "H": [h], "J": [j], "KLD": [kld]})], ignore_index=True)

        def print_results(h, j, kld, h_b, j_b, kld_b):
            print(f"\tAvg:\t\tH: {h:5.2f} \tJ: {j:5.2f} \tKLD: {kld:5.2f}\n\tBest:\t\tH: {h_b:5.2f} \tJ: {j_b:5.2f} \tKLD: {kld_b:5.2f}")
            print()

        def generate_freqdist(references, hypotheses):
            norm = NormModel()
            ref_hyp = references[0] + hypotheses[0]
            ref_hyp_dict = HashDictionary([ref_hyp])
            ref_hyp_bow = ref_hyp_dict.doc2bow(ref_hyp)
            ref_hyp_bow = [(i[0], 0) for i in ref_hyp_bow]
            ref_bow_base = [ref_hyp_dict.doc2bow(text) for text in references][0]
            hyp_bow_base = [ref_hyp_dict.doc2bow(text) for text in hypotheses][0]
            ref_bow, hyp_bow = [], []
            ref_list = [i[0] for i in ref_bow_base]
            hyp_list = [i[0] for i in hyp_bow_base]

            for base in ref_hyp_bow:
                if base[0] not in ref_list:
                    ref_bow.append((base[0], base[1] + 1))
                else:
                    for ref in ref_bow_base:
                        if ref[0] == base[0]:
                            ref_bow.append((ref[0], ref[1] + 1))

            for base in ref_hyp_bow:
                if base[0] not in hyp_list:
                    hyp_bow.append((base[0], base[1] + 1))
                else:
                    for hyp in hyp_bow_base:
                        if hyp[0] == base[0]:
                            hyp_bow.append((hyp[0], hyp[1] + 1))

            ref_bow_norm = norm.normalize(ref_bow)
            hyp_bow_norm = norm.normalize(hyp_bow)
            vec_ref = [i[1] for i in ref_bow_norm]
            vec_hyp = [i[1] for i in hyp_bow_norm]
            return vec_ref, vec_hyp, ref_bow_base, hyp_bow_base

        def run_threads():
            with alive_bar(len(self.candidate_summaries), title="Evaluation with Gensim...", bar=False, spinner="dots") as bar:

                def calculate_indexes(references, hypotheses, h_list, kld_list, j_list, index):
                    h, kld, j = [], [], []
                    for i in range(0, len(references)):
                        (
                            ref_bow_norm,
                            hyp_bow_norm,
                            ref_bow,
                            hyp_bow,
                        ) = generate_freqdist(references, hypotheses)

                        h.append(hellinger(hyp_bow_norm, ref_bow_norm))
                        kld.append(kullback_leibler(hyp_bow_norm, ref_bow_norm))
                        j.append(jaccard(hyp_bow, ref_bow))
                        bar()

                    h_list[index] = h
                    kld_list[index] = kld
                    j_list[index] = j

                prepare_gensim()
                threads = [None] * self.gensim_thread_count
                kld = [None] * self.gensim_thread_count
                j = [None] * self.gensim_thread_count
                h = [None] * self.gensim_thread_count

                for i in range(len(threads)):
                    threads[i] = threading.Thread(target=calculate_indexes, args=(self.references[i], self.hypotheses[i], h, kld, j, i))
                    threads[i].start()

                for i in range(len(threads)):
                    threads[i].join()

                return list(it.chain(*h)), list(it.chain(*kld)), list(it.chain(*j))

        h, kld, j = run_threads()

        h_avg = sum(h) / len(h)
        kld_avg = sum(kld) / len(kld)
        j_avg = sum(j) / len(j)
        h_best = min(h)
        kld_best = min(kld)
        j_best = min(j)

        print_results(h_avg, j_avg, kld_avg, h_best, j_best, kld_best)
        self.results_df = results_concat("Avg", "Gensim", h_avg, j_avg, kld_avg, self.results_df)

    def run_sklearn_eval(self):
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        def prepare_sklearn():
            self.references, self.hypotheses = self.golden_summaries[:], self.candidate_summaries[:]

        def prepare_results_for_csv(data, method, aggregator, metric, cosim):
            return str(data), str(method), str(aggregator), str(metric), "{:5.2f}".format(cosim)

        def results_concat(aggregator, metric, cosim, results_df):
            data, method, a, m, cosim = prepare_results_for_csv(self.data_name, self.method, aggregator, metric, cosim)
            return pd.concat([results_df, pd.DataFrame({"data": [data], "method": [method], "aggregator": [a], "metric": [m], "C": [cosim]})], ignore_index=True)

        def print_results(cosim_avg, cosim_best):
            print(f"\tAvg:\t\tC: {cosim_avg:5.2f}\n\tBest:\t\tC: {cosim_best:5.2f}")
            print()

        cosim = []
        with alive_bar(len(self.candidate_summaries), title="Evaluation with Cosine Similarity...", bar=False, spinner="dots") as bar:
            prepare_sklearn()
            for i in range(0, len(self.hypotheses)):
                Tfidf_vect = TfidfVectorizer()
                vector_matrix = Tfidf_vect.fit_transform([self.hypotheses[i], self.references[i]])
                cosine_similarity_matrix = cosine_similarity(vector_matrix)
                cosim.append(cosine_similarity_matrix[0, 1])
                bar()

        cosim_avg = sum(cosim) / len(cosim)
        cosim_best = max(cosim)
        print_results(cosim_avg, cosim_best)
        self.results_df = results_concat("Avg", "SKLearn", cosim_avg, self.results_df)


if __name__ == "__main__":

    corpora = [
        "cnn_dailymail",
        "big_patent",
        "cnn_corpus_abstractive",
        # "cnn_corpus_extractive",
        "xsum",
        "arxiv_pubmed",
        # "arxiv",
        # "pubmed",
    ]

    summarizers = [
        # "SumyRandom",
        # "SumyLuhn",
        # "SumyLsa",
        # "SumyLexRank",
        # "SumyTextRank",
        # "SumySumBasic",
        # "SumyKL",
        # "SumyReduction",
        # "SumyEdmundson"
        # "Transformers-facebook/bart-large-cnn",
        # "Transformers-google/pegasus-xsum",
        "Transformers-csebuetnlp/mT5_multilingual_XLSum",
    ]

    metrics = [
        # "rouge",
        "gensim",
        # "nltk",
        # "sklearn",
    ]

    ### Running methods and eval locally

    # reader = Data()
    # reader.show_available_databases()
    # for corpus in corpora:
    #     data = reader.read_data(corpus, 100)
    #     method = Method(data, corpus)
    #     method.show_methods()
    #     for summarizer in summarizers:
    #         df = method.run(summarizer)
    #         method.examples_to_csv(0)
    #         evaluator = Evaluator(df, summarizer, corpus)
    #         for metric in metrics:
    #             evaluator.run(metric)
    #             evaluator.metrics_to_csv()
    #         evaluator.join_all_results()

    ### Importing summaries from COLAB data
    # (generated summaries must be in /results/transformers/)

    reader = Data()
    data = reader.read_data("xsum", 5)
    method = Method(data, "xsum")
    for corpus in corpora:
        for summarizer in summarizers:
            df = method.import_summaries(summarizer, corpus)
            evaluator = Evaluator(df, summarizer, corpus)
            for metric in metrics:
                evaluator.run(metric)
                evaluator.metrics_to_csv()
            evaluator.join_all_results()
