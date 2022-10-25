# Metodologia de Síntese Automática de Textos

## Objetivo
Existem em Python várias bibliotecas com implementações dos métodos mais populares para a síntese automática de textos. Neste repositório, apresenta-se uma rotina em Python orientada ao download e leitura de uma seleção de *corpora* relevantes na literatura de Processamento de Linguagem Natural (NLP) e aplicação de uma série de métodos de sumarização estabelecidos, tal qual a avaliação dos resultados gerados.

### Bases de dados disponíveis:
- ArXiv + PubMed
- BIGPATENT
- CNN + Daily Mail
- XSum
- CNN Corpus
- MCTI (Notícias)
- MCTI (Oportunidades)
- MCTI (Políticas)

### Métodos de sumarização extrativa disponíveis:
- Reduction Summarizer
- LexRank Summarizer
- LSA Summarizer
- Luhn Summarizer
- Random Summarizer
- SumBasic Summarizer
- TextRank Summarizer

### Métodos de sumarização abstrativa disponíveis:
- BART
- T5
- PEGASUS

### Métodos de avaliação de sumários disponíveis:
- ROUGE
    - ROUGE-1,2,3,4
    - ROUGE-L
    - ROUGE-W
    - ROUGE-SU-4
- Precision, Recall, F
- Distâncias de Jaccard e Hellinger
- Divergência de Kullback-Leibler
- Similaridade por Cosseno

# Processos
Os processos de importação de dados, sumarização e avaliação estão separados nas classes `Data`, `Method` e `Evaluator`, respectivamente. A função `main` sugerida no script executa e avalia todos os métodos de sumarização extrativa para todas as bases de dados da literatura disponíveis, com um corte de 50 entradas. Este parâmetro, que define o número de sumários $N$ a serem gerados a partir de um dado *corpus* pode ser modificado no método `Data.read_data(corpus, N)`. 

Para a execução sem GPU, sugere-se que os métodos de sumarização sejam executados em baixo volume $(N<100)$, dados os altos tempos de execução, em especial para *transformers* abstrativos. Os resultados apresentados em relatório, gerados com $N=3000$, foram obtidos com a ajuda do Google Colab.

## Resultados
As saídas para cada *corpus* em `.\results` são:
- Todos os sumários gerados em um arquivo com terminação `_examples.csv`
- Todos as métricas para os avaliadores escolhidos em um arquivo com terminação `_results.csv`

Vale ressaltar que as bases da literatura são obtidas pelo pacote `datasets` do `huggingface`, e podem exigir grande espaço em disco.

# Referências 

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