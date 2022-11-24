from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline
from nltk.corpus import stopwords
from collections import Counter
import regex as re
import pandas as pd
import gradio as gr
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")


def run(the_method, text, compression_ratio, use_golden=False, golden=None):
    if the_method[0:4] == "Sumy":
        return run_sumy(the_method, _clean_text(text), compression_ratio), run_eval(use_golden, _clean_text(text), run_sumy(the_method, _clean_text(text), compression_ratio), golden)
    elif the_method[0:13] == "Transformers-":
        return run_transformers(the_method, _clean_text(text), compression_ratio), run_eval(use_golden, _clean_text(text), run_transformers(the_method, _clean_text(text), compression_ratio), golden)

def run_csv(the_method, csv_input, text_column, compression_ratio=1 / 8, use_golden=False):
    df_original = pd.read_csv(csv_input.name)
    text_series = df_original[text_column]
    text_series = text_series.apply(lambda x: _clean_text(x))

    if the_method[0:4] == "Sumy":
        result = run_sumy_df(the_method, text_series, compression_ratio)
        the_method_dir = the_method[4:]
        #run_eval(use_golden, df, run_sumy(the_method, df, compression_ratio))
    elif the_method[0:13] == "Transformers-":
        the_method_dir = re.sub(r"[\/]","-",the_method[13:])
        result = run_transformers_df(the_method, text_series, compression_ratio)
        #run_eval(use_golden, df, run_sumy(the_method, df, compression_ratio))

    column_name = "summary_"+the_method_dir
    df_original[column_name] = result["summary"]
    df_original.to_csv(the_method_dir+"_results.csv", index=False)
    return str(the_method_dir+"_results.csv")


def run_df(the_method, df, compression_ratio=1 / 8, use_golden=False):
    
    text_series = df.iloc[:,0].apply(lambda x: _clean_text(x))
    print(text_series)

    if the_method[0:4] == "Sumy":
        result = run_sumy_df(the_method, text_series, compression_ratio)
        the_method_dir = the_method[4:]
        #run_eval(use_golden, df, run_sumy(the_method, df, compression_ratio))
    elif the_method[0:13] == "Transformers-":
        the_method_dir = re.sub(r"[\/]","-",the_method[13:])
        result = run_transformers_df(the_method, text_series, compression_ratio)
        #run_eval(use_golden, df, run_sumy(the_method, df, compression_ratio))

    result.to_csv(the_method_dir+"_results.csv", index=False)
    return str(the_method_dir+"_results.csv")

def _clean_text(content):
    if isinstance(content, str):
        pass
    else:
        content = str(content)
    # strange jump lines
    content = re.sub(r"\.", ". ", str(content))
    # URLs
    content = re.sub(r"http\S+", "", str(content))
    # trouble characters
    content = re.sub(r"\\r\\n", " ", str(content))
    # clean jump lines
    content = re.sub(r"\u000D\u000A|[\u000A\u000B\u000C\u000D\u0085\u2028\u2029]", " ", content)
    # Replace different spaces
    content = re.sub(r"\u00A0\u1680​\u180e\u2000-\u2009\u200a​\u200b​\u202f\u205f​\u3000", " ", content)
    # replace multiple spaces
    content = re.sub(r" +", " ", content)
    # normalize hiphens
    content = re.sub(r"\p{Pd}+", "-", content)
    # normalize single quotations
    content = re.sub(r"[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]", "'", content)
    # normalize double quotations
    content = re.sub(r"[\u201C-\u201E\u2033\u275D\u275E\u301D\u301E]", '"', content)
    # normalize apostrophes
    content = re.sub(r"[\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "'", content)

    content = " ".join(content.split())
    return content


def run_sumy(method, text, compression_ratio):
    from sumy.summarizers.random import RandomSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.summarizers.sum_basic import SumBasicSummarizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.reduction import ReductionSummarizer
    from sumy.summarizers.edmundson import EdmundsonSummarizer

    the_method = method.replace("Sumy", "")
    summarizer = locals()[the_method + "Summarizer"]()
    sentence_count = int(len(sent_tokenize(text)) * compression_ratio / 100)
    if sentence_count < 1:
        sentence_count = 1
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    summary = summarizer(parser.document, sentence_count)

    text_summary = ""
    for s in summary:
        text_summary += str(s) + " "
    return text_summary


def run_transformers(method, text, compression_ratio):

    the_method = method.replace("Transformers-", "")
    summarizer = pipeline("summarization", model=the_method)

    length = 3000
    while len(word_tokenize(text[0:length])) > 450:
        length -= 100
    token_count = len(word_tokenize(text[0:length])) * compression_ratio / 100
    aux_summary = summarizer(text[0:length], min_length=(int(token_count - 5)), max_length=(int(token_count + 5)))
    summary = aux_summary[0]["summary_text"]
    return summary


def run_sumy_df(method, texts_series, compression_ratio):
    
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

    the_method = method.replace("Sumy", "")
    the_summarizer = locals()[the_method + "Summarizer"]()

    summarizer_output_list = []
    for text in texts_series:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        sentence_count = int(len(sent_tokenize(text)) * compression_ratio / 100)
        if sentence_count < 1:
            sentence_count = 1
        summarizer_output_list.append(the_summarizer(parser.document, sentence_count))

    candidate_summaries = []
    for summarizer_output in summarizer_output_list:
        text_summary = ""
        for sentence in summarizer_output:
            text_summary += str(sentence) + " "

        candidate_summaries.append(text_summary)

    results = pd.DataFrame({"text": texts_series, "summary": candidate_summaries})
    return results

def run_transformers_df(method, texts_series, compression_ratio):
    from transformers import pipeline
    from nltk.tokenize import word_tokenize

    the_method = method.replace("Transformers-", "")
    summarizer = pipeline("summarization", model=the_method)

    aux_summaries_list = []
    for text in texts_series:
        length = 3000
        while len(word_tokenize(text[0:length])) > 450:
            length -= 100
            token_count = len(word_tokenize(text[0:length])) * compression_ratio / 100
        aux_summaries_list.append(summarizer(text[0:length], min_length=int(token_count - 5), max_length=int(token_count + 5)))

    candidate_summaries = [x[0]["summary_text"] for x in aux_summaries_list]

    results = pd.DataFrame({"text": texts_series, "summary": candidate_summaries})
    return results

def run_eval(use_golden, text, summary, golden):
    if use_golden:
        rouge = run_rouge_eval(summary, golden)
        nltk = run_nltk_eval(summary, golden)
        gensim = run_gensim_eval(summary, golden)
        sklearn = run_sklearn_eval(summary, golden)
        return rouge + nltk + gensim + sklearn
    else:
        gensim = run_gensim_eval(summary, text)
        sklearn = run_sklearn_eval(summary, text)
        return gensim + sklearn


def run_rouge_eval(text, golden):
    import rouge
    from rouge_metric import PyRouge

    def print_results(m, p, r, f):
        return str("{}:\t\t{}: {:5.2f} \t{}: {:5.2f} \t{}: {:5.2f}\n".format(str(m).upper(), "P", 100.0 * p, "R", 100.0 * r, "F1", 100.0 * f))

    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l", "rouge-w"],
        max_n=4,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=False,
        apply_best=False,
        alpha=0.5,
        weight_factor=1.2,
        stemming=True,
    )  # Default F1_score

    evaluator_su = PyRouge(
        rouge_n=(1, 2, 3, 4),
        rouge_l=True,
        rouge_w=True,
        rouge_w_weight=1.2,
        # rouge_s=True,
        rouge_su=True,
        skip_gap=4,
    )

    scores = evaluator_su.evaluate([text], [[golden]])

    rouge_strings = ""
    for m, results in sorted(scores.items()):
        p = results["p"]
        r = results["r"]
        f = results["f"]
        rouge_strings += print_results(m, p, r, f)
    return rouge_strings


def run_nltk_eval(text, golden):
    from nltk.metrics.scores import precision, recall, f_measure

    def print_results(p, r, f):
        return str(f"NLTK:\t\t\t\tP: {100*p:5.2f} \tR: {100*r:5.2f} \tF1: {100*f:5.2f}\n")

    p, r, f = [], [], []

    reference = [i for i in golden.split()]
    hypothesis = [i for i in text.split()]

    p = precision(set(reference), set(hypothesis))
    r = recall(set(reference), set(hypothesis))
    f = f_measure(set(reference), set(hypothesis), alpha=0.5)

    return print_results(p, r, f)


def run_gensim_eval(text, golden):
    from gensim.matutils import kullback_leibler, hellinger, jaccard, jensen_shannon
    from gensim.corpora import Dictionary, HashDictionary
    from gensim.models import ldamodel, NormModel

    def print_results(h, j, kld):
        return str(f"Gensim:\t\t\tH: {h:5.2f} \tJ: {j:5.2f} \tKLD: {kld:5.2f}\n")

    def generate_freqdist(text, golden):

        ref_hyp = text + golden
        ref_hyp_dict = HashDictionary([ref_hyp])
        ref_hyp_bow = ref_hyp_dict.doc2bow(ref_hyp)
        ref_hyp_bow = [(i[0], 0) for i in ref_hyp_bow]
        ref_bow_base = [ref_hyp_dict.doc2bow(text) for text in [golden]][0]
        hyp_bow_base = [ref_hyp_dict.doc2bow(text) for text in [text]][0]
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

    ref_bow_norm, hyp_bow_norm, ref_bow, hyp_bow = generate_freqdist(text, golden)

    h = hellinger(hyp_bow_norm, ref_bow_norm)
    kld = kullback_leibler(hyp_bow_norm, ref_bow_norm)
    j = jaccard(hyp_bow, ref_bow)

    return print_results(h, j, kld)


def run_sklearn_eval(text, golden):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    def print_results(cosim_avg):
        return str(f"SKLearn:\t\t\tC: {cosim_avg:5.2f}\n")

    Tfidf_vect = TfidfVectorizer()
    vector_matrix = Tfidf_vect.fit_transform([text, golden])
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    cosim = cosine_similarity_matrix[0, 1]

    return print_results(cosim)


if __name__ == "__main__":

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Sumarização Automática de Textos + Avaliação de Resumos\n Projeto de Pesquisa de Ciência de Dados aplicada ao Portfólio de Produtos Financeiros - PPF-MCTI")
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        dropdown = gr.Dropdown(
                            label="Método de Sumarização",
                            choices=[
                                "SumyRandom",
                                "SumyLuhn",
                                "SumyLsa",
                                "SumyLexRank",
                                # "SumyEdmundson",
                                "SumyTextRank",
                                "SumySumBasic",
                                "SumyKL",
                                "SumyReduction",
                                "Transformers-google/pegasus-xsum",
                                "Transformers-facebook/bart-large-cnn",
                                "Transformers-csebuetnlp/mT5_multilingual_XLSum",
                            ],
                            value="SumyLuhn",
                        )
                    with gr.Column(scale=1, min_width=300):
                        compression_ratio = gr.Slider(
                            label="Taxa de Compressão (% do tamanho original)",
                            value=10,
                            minimum=1,
                            maximum=100,
                        )
                        use_golden = gr.Checkbox(label="Avaliar usando Golden Summary?")
                with gr.Tab("Texto"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=300):
                            text = gr.Textbox(
                                label="Texto",
                                placeholder="Insira seu texto aqui",
                            )
                            golden = gr.Textbox(
                                label="Golden Summary",
                                placeholder="Insira o resumo ideal do texto aqui (opcional)",
                            )
                        with gr.Column(scale=1, min_width=300):
                            generated_summary = gr.Textbox(label="Resumo gerado automaticamente")
                            evaluators = gr.Textbox(label="Avaliação do resumo")
                    text_button = gr.Button("Executar")
                with gr.Tab("CSV"):
                    with gr.Column(scale=1, min_width=300):
                        gr.Checkbox(
                            label="Insira abaixo um arquivo CSV com uma coluna de textos a serem sumarizados. Caso opte por avaliar usando golden summaries, estes deverão estar presentes em uma coluna entitulada 'golden'.",
                            value=False,
                            interactive=False,
                        )
                        with gr.Row():
                            with gr.Column(scale=1, min_width=300):
                                text_column = gr.Textbox(label="Título da coluna a ser sumarizada", placeholder="text")
                                csv_input = gr.File(label="Arquivo .csv de textos")
                            csv_output = gr.Files(label="Arquivos .csv de resumos e avaliação")
                        csv_button = gr.Button("Executar")
                with gr.Tab("DataFrame"):
                    with gr.Column(scale=1, min_width=300):
                        gr.Checkbox(
                            label="Preencha o DataFrame abaixo com textos a serem sumarizados. Caso opte por avaliar usando golden summaries, estes deverão estar presentes na segunda coluna.",
                            value=False,
                            interactive=False,
                        )
                        with gr.Row():
                            df_input = gr.DataFrame(headers=["Texto","Golden Summary"],row_count=(1,"dynamic"),col_count=(2,"fixed"))
                            df_output = gr.Files(label="Arquivos .csv de resumos e avaliação")
                        df_button = gr.Button("Executar")

            text_button.click(run, inputs=[dropdown, text, compression_ratio, use_golden, golden], outputs=[generated_summary, evaluators])
            csv_button.click(run_csv, inputs=[dropdown, csv_input, text_column, compression_ratio, use_golden], outputs=[csv_output])
            df_button.click(run_df, inputs=[dropdown, df_input, compression_ratio, use_golden], outputs=[df_output])

demo.launch()
