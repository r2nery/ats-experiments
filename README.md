# Automatic Text Summarization and Summary Evaluation Experiments

## Reference
This repository is a refactored version of the reproduction package for [A comprehensive review of automatic text summarization techniques: method, data, evaluation and coding](https://doi.org/10.48550/arXiv.2301.03403). Please cite it accordingly.

## Objective
This repository presents Python scripts for downloading and wrangling a selection of relevant *corpora* from the literature of Automatic Text Summarization (ATS), as well as applying a variety of abstractive and extractive methods of summarization. The resulting summaries can then be evaluated using a selection of metrics from the literature. The scripts herein were developed with the processing of entire *corpora* in mind, making use of paralellization. However, single-document summarization and evaluation is also possible.

## CNN Corpus
In addition to popular datasets from the literature on Natural Language Processing (NLP) available for download through the script itself, wrangling functions were implemented for the CNN Corpus dataset, a novel dataset introduced by [Lins, RD et al. (2019](https://doi.org/10.1145/3342558.3345388) that presents both extractive and abstractive reference summaries for CNN News stories. This dataset in particular is included under `.\data`. 

### Available Datasets:
- ArXiv + PubMed
- BIGPATENT
- CNN + Daily Mail
- XSum
- CNN Corpus

### Available Extractive Summarization Methods:
- Reduction Summarizer
- LexRank Summarizer
- LSA Summarizer
- Luhn Summarizer
- Random Summarizer
- SumBasic Summarizer
- TextRank Summarizer

### Available Abstractive Summarization Methods/Checkpoint:
- facebook/bart-large-cnn
- google/pegasus-xsum
- csebuetnlp/mT5_multilingual_XLSum

### Available Summary Evaluation Methods:
- ROUGE
    - ROUGE-1,2,3,4
    - ROUGE-L
    - ROUGE-W
    - ROUGE-SU-4
- Precision, Recall, F1
- Jaccard and Hellinger Divergences
- Kullback-Leibler Divergences
- Cosine Similarity

# Classes and Methods
The data importing, summarization and evaluation processes are defined in the `Data`, `Method` e `Evaluator` classes, respectively. The `main` function suggests the summarization and evaluation of 50 entries from all datasets available. The number of summaries to be generated $N$ can be modified in `Data.read_data(corpus, N)`. 

For CPU workloads, we suggest a low number of summaries $(N<100)$, given the long computing times, especially for abstractive *transformer* models. The results presented on our reports, generated with $N=3000$, were obtained with the help of Google Colab and GPU computing.

## Results
The outputs for each corpus *corpus* in `.\results` are:
- All the summaries generated, in a file terminated by `_examples.csv`
- All the chosen evaluators' results, in a file terminated by `_results.csv`

It should be mentioned that the literature datasets are downloaded using the `datasets` libarary, by huggingface, and may take a large amound of disk space.

# References 

Lins, Rafael Dueire, et al. "The cnn-corpus: A large textual corpus for single-document extractive summarization." Proceedings of the ACM Symposium on Document Engineering 2019. 2019.

Baeza-Yates, R. and Ribeiro-Neto, B. (2008). Modern Information Retrieval, 2nd edn,
Addison-Wesley Publishing Company, USA.

Erkan, G. and Radev, D. R. (2004). Lexrank: Graph-based lexical centrality as salience in text
summarization, Journal Artificial Intelligence Research 22(1): 457–479.

Haghighi, A. and Vanderwende, L. (2009). Exploring content models for multi-document
summarization, Proceedings of human language technologies: The 2009 annual conference of
the North American Chapter of the Association for Computational Linguistics, pp. 362–370.

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and
Zettlemoyer, L. (2019). Bart: Denoising sequence-to-sequence pre-training for natural language
generation, translation, and comprehension, arXiv preprint arXiv:1910.13461 .

Lin, C.-Y. (2004). Rouge: A package for automatic evaluation of summaries, Text summarization
branches out, pp. 74–81.

Louis, A. and Nenkova, A. (2013). Automatically assessing machine summary content without a
gold standard, Computational Linguistics 39(2): 267–300.

Luhn, H. P. (1958). The automatic creation of literature abstracts, IBM Journal of Research and
Development 2(2): 159–165.

Mihalcea, R. and Tarau, P. (2004). Textrank: Bringing order into text, Proceedings of the 2004
conference on empirical methods in natural language processing, pp. 404–411.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu,
P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer,
arXiv preprint arXiv:1910.10683.

Saggion, H., Radev, D. R., Teufel, S., Lam, W. and Strassel, S. M. (2002). Developing infrastructure
for the evaluation of single and multi-document summarization systems in a cross-lingual
environment., LREC, pp. 747–754.

Salton, G. (1989). Automatic text processing, Reading: Addison-Wesley.

Steinberger, J., Jezek, K. et al. (2004). Using latent semantic analysis in text summarization and
summary evaluation, Proceedings of ISIM 4(93-100): 8.

Zhang, J., Zhao, Y., Saleh, M. and Liu, P. (2020). Pegasus: Pre-training with extracted
gap-sentences for abstractive summarization, International Conference on Machine Learning,
PMLR, pp. 11328–11339
