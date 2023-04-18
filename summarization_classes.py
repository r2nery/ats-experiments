# python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 3 2022

@author: Arthur G. Nery, Daniel O. Cajueiro
"""

from rouge_metric import PyRouge
from nltk.metrics.scores import precision, recall, f_measure
from gensim.matutils import kullback_leibler, hellinger, jaccard, jensen_shannon
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import HashDictionary
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from datasets import load_dataset
from transformers import pipeline
import threading
from alive_progress import alive_bar
from bs4 import BeautifulSoup
import pandas as pd
import regex
import os
import re
import itertools as it
import more_itertools as mit
import nltk
import rouge
import shutil

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")

ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))


class Data:
    """
    Data reading and processing class.
    """

    def __init__(self):
        """Establishes ready-to-use corpora."""

        self.available_databases = [
            "CNN_Corpus_Extractive",
            "CNN_Corpus_Abstractive",
            "Opinosis",
            "CNN_DailyMail",
            "Big_Patent",
            "XSum",
            "ArXiv_PubMed",
            "ArXiv",
            "PubMed",
        ]
        self.size = None

    def show_available_databases(self):
        """
        Shows available databases.
        """
        print("\nThe available databases are:")
        for i, database in enumerate(self.available_databases):
            print(str(i) + ": " + database)
        print()

    def wipe_cache(self):
        """
        Wipes cache of huggingface datasets.
        """
        data = os.listdir(os.path.join(ROOT_DIR, "data"))
        for item in data:
            if item.endswith(".lock"):
                os.remove(os.path.join(ROOT_DIR, "data", item))
            if item == "downloads":
                shutil.rmtree(os.path.join(ROOT_DIR, "data", item))

    def clean_text(self, content):
        """
        Removes undesirable characters from text.

        Parameters
        ----------
        content : str
            String of text to be cleaned.

        Returns
        -------
        str
            Cleaned up text.
        """
        if not isinstance(content, str):
            content = str(content)

        # strange jump lines
        content = re.sub(r"\.", ". ", str(content))
        # trouble characters
        content = re.sub(r"\\r\\n", " ", str(content))
        # clean jump lines
        content = re.sub(r"\u000D\u000A|[\u000A\u000B\u000C\u000D\u0085\u2028\u2029]", " ", content)
        # Replace different spaces
        content = re.sub(
            r"\u00A0\u1680​\u180e\u2000-\u2009\u200a​\u200b​\u202f\u205f​\u3000", " ", content
        )
        # replace multiple spaces
        content = re.sub(r" +", " ", content)
        # normalize hiphens
        content = regex.sub(r"\p{Pd}+", "-", content)
        # normalize single quotations
        content = re.sub(r"[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]", "'", content)
        # normalize double quotations
        content = re.sub(r"[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]", '"', content)
        # normalize apostrophes
        content = re.sub(
            r"[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]",
            "'",
            content,
        )

        return content.strip()

    def read_data(self, database_name, size=10000):
        """
        Directs to correct data reading method.

        Parameters
        ----------
        database_name : str
            Corpus name.

        size : int
            Number of entries from corpus to be summarized.

        Returns
        -------
        data : str
            Dataframe with original texts and golden summaries
        """
        self.size = size
        data_reader_func = getattr(self, f"read_{database_name.lower()}")
        data_reader_func()
        self.wipe_cache()
        return self.data

    ### Huggingface Datasets ###

    def read_opinosis(self):
        dataset = load_dataset(
            "opinosis",
            split="train",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
        summaries_dict = {}
        summaries_df = pd.DataFrame(columns=[])
        texts = []
        with alive_bar(len(dataset), title="Loading Opinosis...") as bar:
            for i in range(0, len(dataset)):
                texts.append(" ".join(dataset[i]["review_sents"].split()))
                for j in range(0, len(dataset[i]["summaries"])):
                    summaries_dict["summary " + str(j)] = " ".join(
                        dataset[i]["summaries"][j].split()
                    )
                summaries_df = pd.concat(
                    [summaries_df, pd.DataFrame.from_dict([summaries_dict])], ignore_index=True
                )
                bar()
            texts_df = pd.DataFrame(texts, columns=["text"])
            self.data = pd.concat([texts_df, summaries_df], axis=1)
            self.data = self.data.head(self.size)
            # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    def read_cnn_dailymail(self):
        dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
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
        with alive_bar(
            bar=None, monitor=False, stats=False, spinner="dots", title="Loading BIGPATENT..."
        ):
            dataset = load_dataset(
                "big_patent",
                "all",
                split="test",
                cache_dir=os.path.join(ROOT_DIR, "data"),
            )
            self.data = dataset.to_pandas()
            self.data = self.data.rename(columns={"description": "text", "abstract": "golden"})
            self.data["text"] = self.data["text"].apply(
                lambda x: " ".join(self.clean_text(x).split())
            )
            self.data["golden"] = self.data["golden"].apply(
                lambda x: " ".join(self.clean_text(x).split())
            )
            self.data = self.data.head(self.size)
            # self.data.to_csv(os.path.join(ROOT_DIR,"dataset_test.csv"), index=False)

    def read_xsum(self):
        dataset = load_dataset(
            "xsum",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
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
        pubmed = load_dataset(
            "ccdv/pubmed-summarization",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
        arxiv = load_dataset(
            "ccdv/arxiv-summarization",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
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
        arxiv = load_dataset(
            "ccdv/arxiv-summarization",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
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
        pubmed = load_dataset(
            "ccdv/pubmed-summarization",
            split="test",
            cache_dir=os.path.join(ROOT_DIR, "data"),
        )
        texts, summaries = [], []
        with alive_bar(len(pubmed), title="Loading PubMed...") as bar:
            for i in range(0, len(pubmed)):
                texts.append(" ".join(pubmed[i]["article"].split()))
                summaries.append(" ".join(pubmed[i]["abstract"].split()))
                bar()
        self.data = pd.DataFrame(list(zip(texts, summaries)), columns=["text", "golden"])
        self.data = self.data.head(self.size)
        # self.data.to_csv(os.path.join(ROOT_DIR, "dataset_test.csv"), index=False)

    # CNN Corpus Dataset

    def read_cnn_corpus_abstractive(self):
        """Reads the abstractive portion of CNN Corpus."""

        texts_df = pd.DataFrame(columns=["text"])
        summaries_df = pd.DataFrame()
        text_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "CNN_Corpus")
        processed_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "processed_abstractive")
        texts = os.listdir(text_path)

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

        with alive_bar(len(texts), title="Loading CNN Corpus (Abstractive)") as progress_bar:
            for file in texts:
                with open(os.path.join(text_path, file), "r", encoding="utf8") as f:
                    soup = BeautifulSoup(f.read(), "xml")
                cleanup = {"&quot;": '"', "&apost;": "'"}
                abs_sum = soup.find("highlights").get_text()
                text = soup.find("article").get_text()
                for key, value in cleanup.items():
                    abs_sum = " ".join(abs_sum.replace(key, value).split())
                    text = " ".join(text.replace(key, value).split())
                progress_bar()
                texts_df = pd.concat(
                    [texts_df, pd.DataFrame({"text": [self.clean_text(text)]})],
                    ignore_index=True,
                )
                summaries_df = pd.concat(
                    [summaries_df, pd.DataFrame({"golden": [self.clean_text(abs_sum)]})],
                    ignore_index=True,
                )
            self.data = pd.concat([texts_df, summaries_df], axis=1)
            self.data.to_csv(
                os.path.join(processed_path, "CNN_Corpus_Abstractive.csv"),
                index=False,
            )
        self.data = self.data.head(self.size)

    def read_cnn_corpus_extractive(self):
        """Reads the extractive portion of CNN Corpus."""

        texts_df = pd.DataFrame(columns=["text"])
        summaries_df = pd.DataFrame()
        text_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "CNN_Corpus")
        processed_path = os.path.join(ROOT_DIR, "data", "CNN_CORPUS", "processed_extractive")
        texts = os.listdir(text_path)

        if not os.path.exists(processed_path):
            os.makedirs(processed_path)

        with alive_bar(len(texts), title="Loading CNN Corpus (Extractive)") as progress_bar:
            for file in texts:
                with open(os.path.join(text_path, file), "r", encoding="utf8") as file:
                    soup = BeautifulSoup(file.read(), "xml")
                cleanup = {"&quot;": '"', "&apost;": "'"}
                ext_sum = soup.find("gold_standard").get_text()
                text = soup.find("article").get_text()
                for key, value in cleanup.items():
                    ext_sum = " ".join(ext_sum.replace(key, value).split())
                    text = " ".join(text.replace(key, value).split())
                progress_bar()
                texts_df = pd.concat(
                    [texts_df, pd.DataFrame({"text": [self.clean_text(text)]})],
                    ignore_index=True,
                )
                summaries_df = pd.concat(
                    [summaries_df, pd.DataFrame({"golden": [self.clean_text(ext_sum)]})],
                    ignore_index=True,
                )
            self.data = pd.concat([texts_df, summaries_df], axis=1)
            self.data.to_csv(
                os.path.join(processed_path, "CNN_Corpus_Extractive.csv"),
                index=False,
            )
        self.data = self.data.head(self.size)


class Method:
    """
    Corpus summarizator class.
    """

    def __init__(self, data_df, data_name):
        """
        Establishes ready-to-use summarizers. Sets up dataset for summarization.
        """
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
            "cnn_corpus_abstractive": [3, 55],
            "cnn_corpus_extractive": [4, 150],
            "opinosis": [1, 30],
            "cnn_dailymail": [2, 40],
            "big_patent": [4, 130],
            "xsum": [1, 20],
            "arxiv_pubmed": [6, 240],
            "arxiv": [6, 240],
            "pubmed": [6, 240],
        }
        self.sentence_count, self.token_count = self.target_lengths[data_name]

    def show_methods(self):
        """
        Prints out available summarization models.
        """
        print("\nThe available summarization methods are:")
        for i, model in enumerate(self.available_methods):
            print(f"{i}: {model}")
        print()

    def examples_to_csv(self, size=10000):
        """
        Saves a number of generated summaries to a CSV file.

        Parameters
        ----------
        size : int
            Number of lines exported to CSV file.
        """
        path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{self.data_name}_examples.csv"
        if not os.path.exists(os.path.join(path, filename)):
            self.results.head(size).to_csv(os.path.join(path, filename), index=False)
        else:
            old = pd.read_csv(os.path.join(path, filename))
            new = pd.concat([old, self.results.head(size)]).drop_duplicates()
            new.to_csv(os.path.join(path, filename), index=False)

    def run(self, the_method):
        """
        Directs to either sumy or transformer summarization.

        Parameters
        ----------
        the_method : str
            Chosen summarization model.
        """
        self.the_method = the_method
        if self.the_method.startswith("Sumy"):
            self.run_sumy()
        elif self.the_method.startswith("Transformers-"):
            self.run_transformers()
        else:
            print("This method is not defined! Try another one.")
        print(f"{len(self.candidate_summaries)} Summaries generated.\n")
        return self.results

    def run_sumy(self):
        """
        Runs extractive summarization.

        Returns
        -------
        results : dataframe
            Dataframe containing method used, original texts, generated summaries and golden summaries
        """

        the_method = self.the_method.replace("Sumy", "")
        the_summarizer = globals()[the_method + "Summarizer"]()

        with alive_bar(
            len(self.texts),
            bar=None,
            spinner="dots",
            title=f"Running {self.the_method} Summarizer",
        ) as progress_bar:
            summarizer_output_list = []
            for text in self.texts:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summarizer_output_list.append(the_summarizer(parser.document, self.sentence_count))
                progress_bar()

        self.candidate_summaries = [
            " ".join(str(sentence) for sentence in summarizer_output)
            for summarizer_output in summarizer_output_list
        ]

        self.results = pd.DataFrame(
            {
                "method": [self.the_method] * len(self.candidate_summaries),
                "summary": self.candidate_summaries,
                "golden": self.golden_summaries,
                "source": self.texts,
            }
        )

    def run_transformers(self):
        """
        Runs abstractive summarization.

        Returns
        -------
        results : dataframe
            Dataframe containing method used, original texts, generated summaries and golden summaries
        """

        the_method = self.the_method.replace("Transformers-", "")
        with alive_bar(
            len(self.texts), bar=None, spinner="dots", title="Running Transformers-" + the_method
        ) as progress_bar:
            summarizer = pipeline("summarization", model=the_method)

            self.aux_summaries_list = []
            for text in self.texts:
                length = 3000
                while len(word_tokenize(text[0:length])) > 450:
                    length -= 100
                self.aux_summaries_list.append(
                    summarizer(
                        text[0:length],
                        min_length=(self.token_count - 5),
                        max_length=(self.token_count + 5),
                    )
                )
                progress_bar()

        self.candidate_summaries = [x[0]["summary_text"] for x in self.aux_summaries_list]

        self.results = pd.DataFrame(
            {
                "method": self.the_method,
                "summary": self.candidate_summaries,
                "golden": self.golden_summaries,
                "source": self.texts,
            }
        )


class Evaluator:
    """
    Summary evaluator class.
    """

    def __init__(self, data_df, method, data_name):
        """Establishes ready-to-use evaluators. Sets up summaries for evaluation."""
        self.golden_summaries = data_df["golden"].tolist()
        self.candidate_summaries = data_df["summary"].to_list()
        self.available_evaluators = ["rouge", "nltk", "gensim", "sklearn"]
        self.method = method
        self.data_name = data_name
        self.results_df = pd.DataFrame(
            columns=["data", "method", "aggregator", "metric", "P", "R", "F1", "H", "J", "KLD", "C"]
        )

    def show_evaluators(self):
        """Prints out available evaluators."""
        print("The avaliable evaluators are:")
        for i, evaluator in enumerate(self.available_evaluators):
            print(str(i) + ": " + evaluator)
        print()

    def run(self, the_evaluator):
        """
        Directs to the correct evaluator.

        Parameters
        ----------
        the_evaluator : str
            Chosen evaluation method.
        """
        self.the_evaluator = the_evaluator
        evaluators = {
            "rouge": self.run_rouge_eval,
            "nltk": self.run_nltk_eval,
            "gensim": self.run_gensim_eval,
            "sklearn": self.run_sklearn_eval,
        }
        if the_evaluator not in evaluators:
            print(f"This evaluator ({the_evaluator}) is not defined! Try another one.")
            return
        evaluators[the_evaluator]()

    def metrics_to_csv(self):
        """Exports the results dataframe to csv."""
        results_path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        file_path = os.path.join(results_path, f"{self.data_name}_results.csv")
        if os.path.exists(file_path):
            old_results = pd.read_csv(file_path)
            new_results = pd.concat([old_results, self.results_df]).drop_duplicates()
            new_results.to_csv(file_path, index=False)
        else:
            self.results_df.to_csv(file_path, index=False)

    def join_all_results(self):
        """
        Concatenates new results to previous results csv.
        """
        results_dir = os.path.join(ROOT_DIR, "results")
        results_files = [
            f
            for f in os.listdir(results_dir)
            if f.endswith("_results.csv") and not f.endswith("all_results.csv")
        ]
        all_results_path = os.path.join(results_dir, "all_results.csv")
        dfs_to_concat = [pd.read_csv(os.path.join(results_dir, f)) for f in results_files]
        join_df = pd.concat(dfs_to_concat).drop_duplicates()
        join_df.to_csv(all_results_path, index=False)

    def run_rouge_eval(self):
        """Runs ROUGE evaluators."""

        def prepare_rouge():
            self.references = []
            self.hypotheses = self.candidate_summaries
            for golden_summary in self.golden_summaries:
                self.references.append([golden_summary])

        def prep_results_for_csv(data, method, agg, metric, p, r, f):
            fmt = lambda a: f"{100 * a:5.2f}"
            return (str(data), str(method), str(agg), str(metric), fmt(p), fmt(r), fmt(f))

        def results_concat(aggregator, metric, results, results_df):
            data, method, a, m, p, r, f = prep_results_for_csv(
                self.data_name,
                self.method,
                aggregator,
                metric,
                results["p"],
                results["r"],
                results["f"],
            )
            return pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "data": [data],
                            "method": [method],
                            "aggregator": [a],
                            "metric": [m],
                            "P": [p],
                            "R": [r],
                            "F1": [f],
                        }
                    ),
                ],
                ignore_index=True,
            )

        def print_res(metric, p, r, f):
            return f"\t{metric}:\tP: {100 * p:5.2f}\tR: {100 * r:5.2f}\tF1: {100 * f:5.2f}"

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
                rouge_su=True,
                skip_gap=4,
            )

            with alive_bar(
                bar=False,
                monitor=False,
                stats=False,
                spinner="dots",
                title="Evaluation with ROUGE...",
            ) as progress_bar:
                scores = evaluator.get_scores(self.hypotheses, self.references)
                if apply_avg:
                    su = evaluator_su.evaluate(self.hypotheses, self.references)
                    scores = dict(scores, **su)

            print(f"\t{aggregator}:")

            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                if not apply_avg and not apply_best:
                    for hyp, r in enumerate(results):
                        nb_references = len(r["p"])
                        for ref in range(nb_references):
                            print(f"\tHypothesis #{hyp} & Reference #{ref}: ")
                            print(f"\t{print_res(metric, r['p'][ref], r['r'][ref], r['f'][ref])}")
                    print()
                else:
                    print(print_res(metric, results["p"], results["r"], results["f"]))
                    self.results_df = results_concat(aggregator, metric, results, self.results_df)
                    # results_concat(aggregator, metric, results, results_df, data_name, method):
            print()

    def run_nltk_eval(self):
        """Runs NLTK evaluators. (Precision, Recall, F-measure)"""

        def prepare_nltk():
            self.references = [summary.split() for summary in self.golden_summaries]
            self.hypotheses = [summary.split() for summary in self.candidate_summaries]

        def prep_results_for_csv(data, method, agg, metric, p, r, f):
            fmt = lambda a: f"{100 * a:5.2f}"
            return (str(data), str(method), str(agg), str(metric), fmt(p), fmt(r), fmt(f))

        def results_concat(aggregator, metric, precision, recall, fmeasure, results_df):
            data, method, a, m, p, r, f = prep_results_for_csv(
                self.data_name,
                self.method,
                aggregator,
                metric,
                precision,
                recall,
                fmeasure,
            )
            return pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "data": [data],
                            "method": [method],
                            "aggregator": [a],
                            "metric": [m],
                            "P": [p],
                            "R": [r],
                            "F1": [f],
                        }
                    ),
                ],
                ignore_index=True,
            )

        def print_res(p, r, f, p_m, r_m, f_m):
            fmt = lambda a: f"{100 * a:5.2f}"
            print(
                f"\tAvg:\t\tP: {fmt(p)} \tR: {fmt(r)} \tF1: {fmt(f)}\n"
                f"\tBest:\t\tP: {fmt(p_m)} \tR: {fmt(r_m)} \tF1: {fmt(f_m)}\n"
            )

        p, r, f = [], [], []
        with alive_bar(
            len(self.candidate_summaries),
            title="Evaluation with NLTK...",
            bar=False,
            spinner="dots",
        ) as progress_bar:
            prepare_nltk()
            for i in range(0, len(self.hypotheses)):
                p.append(precision(set(self.references[i]), set(self.hypotheses[i])))
                r.append(recall(set(self.references[i]), set(self.hypotheses[i])))
                f.append(f_measure(set(self.references[i]), set(self.hypotheses[i]), alpha=0.5))
                progress_bar()
            p_avg = sum(p) / len(p)
            r_avg = sum(r) / len(r)
            f_avg = sum(f) / len(f)
            p_best = max(p)
            r_best = max(r)
            f_best = max(f)

        print_res(p_avg, r_avg, f_avg, p_best, r_best, f_best)
        self.results_df = results_concat("Avg", "NLTK", p_avg, r_avg, f_avg, self.results_df)

    def run_gensim_eval(self):
        """Runs Gensim evaluators. (Hellinger, Jaccard, Kullback-Leibler)"""
        self.gensim_threads = 10000

        def prepare_gensim():
            self.references = [ref.split() for ref in self.golden_summaries]
            self.hypotheses = [hyp.split() for hyp in self.candidate_summaries]
            self.hypotheses = [list(x) for x in mit.divide(self.gensim_threads, self.hypotheses)]
            self.references = [list(x) for x in mit.divide(self.gensim_threads, self.references)]

        def prep_results_for_csv(data, method, agg, metric, h, j, kld):
            fmt = lambda a: f"{a:5.2f}"
            return (str(data), str(method), str(agg), str(metric), fmt(j), fmt(h), fmt(kld))

        def results_concat(aggregator, metric, h, j, kld, results_df):
            data, method, a, m, j, h, kld = prep_results_for_csv(
                self.data_name, self.method, aggregator, metric, h, j, kld
            )
            return pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "data": [data],
                            "method": [method],
                            "aggregator": [a],
                            "metric": [m],
                            "H": [h],
                            "J": [j],
                            "KLD": [kld],
                        }
                    ),
                ],
                ignore_index=True,
            )

        def print_res(h, j, kld, h_b, j_b, kld_b):
            print(
                f"\tAvg:\t\tH: {h:5.2f} \tJ: {j:5.2f} \tKLD: {kld:5.2f}\n"
                f"\tBest:\t\tH: {h_b:5.2f} \tJ: {j_b:5.2f} \tKLD: {kld_b:5.2f}\n"
            )

        def generate_freqdist(references, hypotheses):
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

            sum_ref = sum([i[1] for i in ref_bow])
            sum_hyp = sum([i[1] for i in ref_bow])
            vec_ref = [i[1] / sum_ref for i in ref_bow]
            vec_hyp = [i[1] / sum_hyp for i in hyp_bow]

            return vec_ref, vec_hyp, ref_bow_base, hyp_bow_base

        def run_threads():
            with alive_bar(
                len(self.candidate_summaries),
                title="Evaluation with Gensim...",
                bar=False,
                spinner="dots",
            ) as progress_bar:

                def calculate_indexes(references, hypotheses, h_list, kld_list, j_list, index):
                    h, kld, j = [], [], []
                    for _ in range(len(references)):
                        (
                            ref_bow_norm,
                            hyp_bow_norm,
                            ref_bow,
                            hyp_bow,
                        ) = generate_freqdist(references, hypotheses)

                        h.append(hellinger(hyp_bow_norm, ref_bow_norm))
                        kld.append(kullback_leibler(hyp_bow_norm, ref_bow_norm))
                        j.append(jaccard(hyp_bow, ref_bow))
                        progress_bar()

                    h_list[index] = h
                    kld_list[index] = kld
                    j_list[index] = j

                prepare_gensim()
                threads = [None] * self.gensim_threads
                kld = [None] * self.gensim_threads
                j = [None] * self.gensim_threads
                h = [None] * self.gensim_threads

                for i in range(len(threads)):
                    threads[i] = threading.Thread(
                        target=calculate_indexes,
                        args=(self.references[i], self.hypotheses[i], h, kld, j, i),
                    )
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

        print_res(h_avg, j_avg, kld_avg, h_best, j_best, kld_best)
        self.results_df = results_concat("Avg", "Gensim", h_avg, j_avg, kld_avg, self.results_df)

    def run_sklearn_eval(self):
        """Runs Scikit-Learn evaluators. (Cosine Similarity)"""

        def prepare_sklearn():
            self.references = self.golden_summaries[:]
            self.hypotheses = self.candidate_summaries[:]

        def prep_results_for_csv(data, method, agg, metric, cosim):
            return str(data), str(method), str(agg), str(metric), f"{100 * cosim:5.2f}"

        def results_concat(aggregator, metric, cosim, results_df):
            data, method, a, m, cosim = prep_results_for_csv(
                self.data_name, self.method, aggregator, metric, cosim
            )
            new_row = pd.DataFrame(
                {"data": [data], "method": [method], "aggregator": [a], "metric": [m], "C": [cosim]}
            )
            return pd.concat([results_df, new_row], ignore_index=True)

        def print_res(cosim_avg, cosim_best):
            print(f"\tAvg:\t\tC: {cosim_avg:5.2f}\n\tBest:\t\tC: {cosim_best:5.2f}\n")

        cosim = []
        with alive_bar(
            len(self.candidate_summaries),
            title="Evaluation with Cosine Similarity...",
            bar=False,
            spinner="dots",
        ) as progress_bar:
            prepare_sklearn()
            for i in range(len(self.hypotheses)):
                Tfidf_vect = TfidfVectorizer()
                vector_matrix = Tfidf_vect.fit_transform([self.hypotheses[i], self.references[i]])
                cosine_similarity_matrix = cosine_similarity(vector_matrix)
                cosim.append(cosine_similarity_matrix[0, 1])
                progress_bar()
        cosim_avg = sum(cosim) / len(cosim)
        cosim_best = max(cosim)
        print_res(cosim_avg, cosim_best)
        self.results_df = results_concat("Avg", "SKLearn", cosim_avg, self.results_df)


if __name__ == "__main__":
    # SIZE OF ALL DATASETS: 38,04 GB
    # UP TO 60 GB DURING DOWNLOAD!
    # SIZE OF CNN CORPUS: 39,9 MB
    # Known conflicts:
    # arxiv,pubmed,arxiv+pubmed + pegasus-xsum

    corpora = [
        "cnn_corpus_abstractive",
        "cnn_corpus_extractive",
        # "cnn_dailymail",
        # "big_patent",
        # "xsum",
        # "pubmed",
        # "arxiv",
        # "arxiv_pubmed",
    ]

    summarizers = [
        "SumyRandom",
        "SumyLuhn",
        "SumyLsa",
        "SumyLexRank",
        "SumyTextRank",
        "SumySumBasic",
        "SumyKL",
        "SumyReduction",
        # "Transformers-facebook/bart-large-cnn",
        # "Transformers-google/pegasus-xsum",
        # "Transformers-csebuetnlp/mT5_multilingual_XLSum",
    ]

    metrics = [
        "rouge",
        "gensim",
        "nltk",
        "sklearn",
    ]

    # Batch summarization
    # Beware of runtime for transformers at high sizes!

    reader = Data()
    reader.show_available_databases()
    for corpus in corpora:
        data = reader.read_data(corpus, size=1)  # size = number of elements from corpus
        method = Method(data, corpus)
        method.show_methods()
        for summarizer in summarizers:
            df = method.run(summarizer)
            method.examples_to_csv(size=0)
            evaluator = Evaluator(df, summarizer, corpus)
            for metric in metrics:
                evaluator.run(metric)
                evaluator.metrics_to_csv()
            evaluator.join_all_results()
