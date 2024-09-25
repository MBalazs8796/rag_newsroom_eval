import pandas as pd
import json
import numpy as np
from openai import OpenAI
from pydantic import BaseModel


def Propertional_sample(df , total_samples):
    counts = df['Topic'].value_counts()


    proportions = counts / counts.sum()
    samples_per_conflict = (proportions * total_samples).round().astype(int)
    sampled_dfs = []
    for file, num_samples in samples_per_conflict.items():
        sampled_df = df[df['Topic'] == file].sample(n=num_samples, random_state=1)
        sampled_dfs.append(sampled_df)

    Question_df = pd.concat(sampled_dfs, ignore_index=True)
    return Question_df


cats = ['Event-based Questions', 'Location-based Questions', 'Causal Questions', 'People/Actor-based Questions', 'Temporal Questions', 'Quantitative Questions', 'Procedural Questions', 'Comparative Questions']

def Generate_question(Article, category, num_generations=1):
    prompt = f"""
        You are given an article related to a conflict. Your task is to generate a set of challenging multiple choice questions based on the article in each of the following categories (if possible). the question must be meaningful and make sense. If in a category you can not generate such a question just leave it:

        {category}


            For the article provided, follow these steps:
            1.	Generate a multiple choice question with 4 choices that can be answered using the article. Generate a clear, concise question based on the article, ensuring it fits one of the categories above. Aim for brevity in both the question and the answer choices. Avoid using vague time references such as "recently," "currently," or "last week", "Monday" etc. and avoid ambiguous pronouns like "he," "they," etc.
            Each question should follow this format:
                    Question: [One single question here including four options labeled a, b, c, d]
                    a) [First option]
                    b) [Second option]
                    c) [Third option]
                    d) [Fourth option]

            2.	Provide the correct answer to the question, based solely on the article's content.
            3.	Identify the category that the question falls under.

            Output Format:
            •	Multiple choice question: [Generated question + 4 choices]
            •	Correct Answer: [The correct answer based on the article, either A,B,C or D]
            •	Category: [One of the eight categories listed above]

            considerations: It would be best if you don't use the exact words in the article text. if possible, use proper synonyms.


            Article: {Article}

    """

    client = OpenAI(
        api_key="")

    class Question(BaseModel):
        MultiplechoiseQuestion: str
        CorrectAnswer: str
        Category: str

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
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
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format=Question
    )
    return response.choices[0].message.content

def local_question_generator(df):

        Question_df = Propertional_sample(df, total_samples=3)

        rows = []

        for index, row in Question_df.iterrows():
            article_text = row['Summary']
            Topic = row['Topic']


            Q = [Generate_question(article_text, cats[i], num_generations=1) for i in range(len(cats))]


            for question_data_json in Q:
                questions_data = json.loads(question_data_json)

                rows.append({
                        'Question': questions_data['MultiplechoiseQuestion'],
                        'Answer': questions_data['CorrectAnswer'],
                        'Category': questions_data['Category'],
                        'Article': article_text,
                        'Topic': Topic
                    })

        questions_df = pd.DataFrame(rows)
        questions_df.to_csv('questions_dataframe.csv', index=False)
        print(questions_df)


df = pd.read_csv('news_dataset.csv')
local_question_generator(df)



