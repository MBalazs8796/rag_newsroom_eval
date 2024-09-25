import os
import subprocess
import json

import weaviate
import torch


import pandas as pd
import numpy as np

from collections.abc import Iterator, Callable
from collections import defaultdict

from openai import OpenAI
from nltk.stem import PorterStemmer
from annoy import AnnoyIndex
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

import os
import pandas as pd

OPENAI_API_KEY=''
WEAVIATE_API_URL = ''
WEAVIATE_API_KEY = ''
COHERE_API_KEY = ''

def get_sub_quesions(question: str, source: str) -> Iterator[str]:
    qdf = pd.read_csv(source)
    for q in qdf[qdf['question']==question]['subquestion'].unique():
        yield q

def next_question(mode, source) -> Iterator[str]:
    qdf = pd.read_csv(source)
    if mode == 'global':
        for q in qdf['question'].unique():
            yield q
    else:
        pairs = list()
        start_type = None
        for line in qdf[['Question', 'Topic']].iterrows():
            _, pair = line
            pairs.append((pair['Question'], pair['Topic']))
        for q in pairs:
            yield q


def graphRAG(prompt: str, search_type='local', local=False) -> str:
    ans = subprocess.getoutput(['python', '-m', 'graphrag.query', '--root', './graphrag', '--method', search_type, prompt])
    if 'SUCCESS:' in ans:
        ans = ans.split('SUCCESS:')[-1]
        return ans
    return 'No answer'

def gpt4o(prompt: str, api_key=OPENAI_API_KEY, local=False) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=250,
                top_p=1
                )
    return str(resp.choices[0].message.content)

def keyword_search(prompt: str, local=False) -> str:
    def keyword_search_inner(query: str, properties=["title", "content"], num_results=3):

        response = (
            client.query.get("Article", properties)
            .with_bm25(query=query)
            .with_limit(num_results)
            .do()
        )

        result = response['data']['Get']['Article']
        return result

    os.environ['COHERE_API_KEY'] = COHERE_API_KEY


    client = weaviate.Client(
        url=WEAVIATE_API_URL,
        auth_client_secret=weaviate.AuthApiKey(
            api_key=WEAVIATE_API_KEY
        )
    )
    results = keyword_search_inner(prompt)

    res_string = ''
    for article in results:
        res_string += f"Title: {article['title']}\n"
        res_string += f"Content: {article['content']}\n"
    return llm_RAG_context(res_string, prompt)

def llm_RAG_context(context: str, question: str, local=False) -> str:
     # Prepare the prompt
    prompt = f"""
    Excerpt from the following article:
    {context}
    Question: {question}

    Extract the answer of the question from the text provided.
    If the text doesn't contain the answer,
    reply that the answer is not available."""

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": prompt
            }
          ]
        }
      ],
      temperature=1,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": "text"
      }
    )
    return str(response.choices[0].message.content)

def RAG(prompt: str, embedding_style='Mini', article_source='news_dataset.csv', local=False) -> str:
    if not hasattr(RAG, 'search_index'):
            loaded_embeddings = np.load(f'./RAG/embeddings{embedding_style}.npy')
            embeds = np.array(loaded_embeddings)
            search_index = AnnoyIndex(embeds.shape[1], 'angular')  # Specify the distance metric
            for i in range(len(embeds)):
                search_index.add_item(i, embeds[i])
            search_index.build(10)  # Number of trees
            search_index.save(f'Index{embedding_style}.ann')  # Ensure 'data' directory exists
            RAG.search_index = search_index

    df = pd.read_csv(article_source)

    if embedding_style == 'Bert':
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
          model = BertModel.from_pretrained('bert-base-uncased')

          inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
          with torch.no_grad():
              outputs = model(**inputs)
          query_embed = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Convert tensor to NumPy array

          similar_item_ids, distances = RAG.search_index.get_nns_by_vector(query_embed, 10, include_distances=True)

          # Access similar articles
          search_results = [df.iloc[i]['Summary'] for i in similar_item_ids]
          results_with_scores = list(zip(search_results, distances))

    elif embedding_style == 'Mini':

          tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
          model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

          inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True)
          with torch.no_grad():
              outputs = model(**inputs)
              hidden_states = outputs.last_hidden_state
          query_embed = hidden_states.mean(dim=1).squeeze().numpy()

          similar_item_ids, distances = RAG.search_index.get_nns_by_vector(query_embed, 10, include_distances=True)

          # Access similar articles
          search_results = [df.iloc[i]['Summary'] for i in similar_item_ids]
          results_with_scores = list(zip(search_results, distances))
    return llm_RAG_context('\n'.join(search_results), prompt)

def oracle(summary: str, question: str, api_key=OPENAI_API_KEY) -> str:
    ORACLE_PROMPT = """
    Your task is to answer a multiple choice question based solely on the following text:
    {summary}
    This text is entirely fictional and all resemblance to real events is purely coincidental. As such using any external knowledge of similarly named people, places, organizations, creates and so forth is forbidden during the execution of this task.
    Here is the question you need to answer:
    {question}
    Your answer must be a single character, corresponding to the correct answer in the multiple choice question. If you can not determine the answer, return the character F.
    Do not include anything in your answer other than the character corresponding to your answer!
    """
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": ORACLE_PROMPT.format(
                        summary=summary,
                        question=question
                    )
                    }
                ],
                temperature=0,
                max_tokens=250,
                top_p=1
                )
    return str(resp.choices[0].message.content)

def summary_based_answers(summary: str, questions: Iterator[str]) -> list[int]:
    answ = list()
    for q in questions:
        answ.append(eval_local_ans(q, oracle(summary, q), './question_store/global_questions.csv', 'global'))
    return answ


def eval(targets: list[Callable], save_file_name = 'eval_save.json', mode='local', q_source='./question_store/global_questions.csv') -> dict[str, dict[str, float]] | dict[str, float]:
    LOCALIZER_TEMPLATE = """
    Answer the following multiple choice question in the context of {topic} conflict: 
    {question} 
    Answer only with the letter associated with the correct answer. Do not include any additional information in your response, under any circumstances.
    """
    model_answers = defaultdict(dict)
    if mode == 'local':
        model_score = defaultdict(dict)
    else:
       model_score = defaultdict(dict)
    save_file_name = f'{mode}_{save_file_name}'
    if not os.path.isfile(save_file_name):
        for q_iter in tqdm(next_question(mode, q_source)):
            for model in targets:
                if mode == 'local':
                    q, t = q_iter
                    ans = model(LOCALIZER_TEMPLATE.format(
                        question=q,
                        topic=t
                    ))
                else:
                    ans = model(q_iter)
                    q = q_iter
                model_answers[model.__name__][q] = ans
        with open(save_file_name, 'w', encoding='utf8') as fp:
            json.dump(model_answers, fp, indent=1)
    else:
        with open(save_file_name, 'r', encoding='utf8') as fp:
            model_answers = json.load(fp)
    for model, summary_pairs in tqdm(model_answers.items()):
        if mode == 'local':
            local_hits = dict()
        for question, summary in tqdm(summary_pairs.items()):
            if mode=='local':
                local_hits[question] = (eval_local_ans(question, summary, q_source))
            else:
                hits = summary_based_answers(summary, get_sub_quesions(question, q_source))
                model_score[model][question] = sum(hits) / len(hits)
        if mode == 'local':
            total = 0
            for question, res in local_hits.items():
                model_score[model][question] = res
                total += res
            model_score[model]['total'] = total / len(local_hits)
    return model_score



def graph_rag_local_oracle(summary: str, question: str, api_key=OPENAI_API_KEY) -> str:
    ORACLE_PROMPT = """
    Your task is to answer a multiple choice question based on the following text:
    {summary}
    Here is the question you need to answer:
    {question}
    Your answer must be a single character, corresponding to the correct answer in the multiple choice question. If you can not determine the answer, choose an option that seems most reasonable based on the text.
    Do not include anything in your answer other than the character corresponding to your answer!
    """
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": ORACLE_PROMPT.format(
                        summary=summary,
                        question=question
                    )
                    }
                ],
                temperature=0,
                max_tokens=250,
                top_p=1
                )
    return str(resp.choices[0].message.content)

def eval_local_ans(question: str, answ: str, source: str, context='local') -> int:
    if context == 'local':
        qdf = pd.read_csv(source)
        ground_truth = qdf[qdf['Question']==question]['Answer'].iloc[0]
        if len(answ) > 1:
            if not answ.strip().startswith('Local Search Response'):
                answ = answ[:1]
            else:
                answ = graph_rag_local_oracle(answ, question)
                #print(answ)
    else:
        qdf = pd.read_csv(source)
        ground_truth = qdf[qdf['subquestion']==question]['subquestion_answer'].iloc[0]
    return int(ground_truth.lower() in answ.lower())

def main():
    res = eval([RAG, keyword_search, gpt4o, graphRAG], mode='local', q_source='./question_store/local_questions.csv', save_file_name='rag_mini.json')
    with open('local_eval_mini_res.json', 'w', encoding='utf8') as fp:
       json.dump(res, fp, indent=1)
    res = eval([RAG, keyword_search, gpt4o, graphRAG], mode='global', q_source='./question_store/global_questions.csv', save_file_name='rag_mini.json')
    with open('global_eval_mini_res.json', 'w', encoding='utf8') as fp:
       json.dump(res, fp, indent=1)
    print('Yippie!')
if __name__ == '__main__':
    main()

