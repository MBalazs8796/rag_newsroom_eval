from openai import OpenAI
from pydantic import BaseModel

def Generate_question(Article, QperArticle, num_generations=1):

    prompt = f"""
    You are a helpful assistant tasked with extracting {QperArticle} questions along with their answers from a news article I will provide. Ensure that each answer is directly found in the article.
    Avoid generating questions with vague terms like "recent" or "latest", but it is allowed to use exact points in time like "2024" or "2023-01-01" etc.
    Follow this format:

    Question: [One single question here]
    Answer: [The smallest exact segment of the article that contains the answer]

    Here is the news article:

    News article: {Article}.
    """

    client = OpenAI(api_key="")

    class Question(BaseModel):
        Questions: list[str]
        Answers: list[str]

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
      response_format= Question
    )
    return response.choices[0].message.content


Article = '''
Russia is withdrawing 100 of its paramilitary officers from Burkina Faso to help in the war in Ukraine.
They are part of about 300 soldiers from the Bear Brigade - a Russian private military company - who arrived in the West African nation in May to support the country's military junta.
On its Telegram channel, the group said its forces would return home to support Russia’s defence against Ukraine’s recent offensive in the Kursk region.
There are fears the pull-out could embolden Islamist insurgents in Burkina Faso, who recently killed up to 300 people in one of the biggest attacks in years.
Burkina Faso has since 2015 suffered regular jihadist attacks, with more than two million people displaced in what aid groups call the world’s “most neglected” crisis.
How Russia has rebranded Wagner in Africa
Junta chiefs 'turn their backs' on West Africa bloc
The junta under interim President Capt Ibrahim Traoré, who came to power in a coup in September 2022, promised to end the attacks but has struggled, even after seeking new security partnerships with Russia.
With nearly half the country outside government control, jihadist groups are increasingly targeting civilians and military units.
Survivors say up to 300 people were killed on Saturday in the northern town of Barsalogho, in an attack which was claimed by an al-Qaeda-linked armed group, Jamaat Nusrat al-Islam wal-Muslimin (JNIM).
They were reportedly both civilians and military personnel helping to dig trenches to help protect the town against jihadist attacks.
The authorities have not said how many people were killed but Communication Minister Rimtalba Jean Emmanuel Ouedraogo called the attack “barbaric”.
The Bear Brigade is said to be responsible for guarding senior Burkinabè officials, including Capt Traoré, whose leadership has been threatened before.
They arrived in the same month when gunshots were fired in the Burkinabè capital near the presidential palace, heightening speculation about growing opposition to the junta leader, who claimed to have thwarted a coup attempt last year.
Videos which circulated on social media and reportedly confirmed by the group showed the Burkinabè military leader being guarded by men in uniforms featuring Russian flags.
The group says it is guarding the Russian ambassador in Ouagadougou, the Burkina Faso capital.
About 100 members of this specialised unit are set to leave the West African country, only three months after arriving.
Their sudden departure is linked to the recent Ukrainian offensive in Russia’s Kursk region.
"When the enemy arrives on our Russian territory, all Russian soldiers forget about internal problems and unite against a common enemy," Bears Brigade commander Viktor Yermolaev told France's Le Monde newspaper (in French).
On Tuesday, the group posted on its Telegram channel that the unit was returning to its base in Russian-occupied Crimea "in connection with recent events."
It is not clear how the Burkina Faso junta plans to compensate for the loss of military support after the partial withdrawal of the Bear Brigade.
Burkina's Faso, like its neighbours, Mali and Niger, is battling various Islamist groups, which operate in the semi-arid Sahel region, south of the Sahara Desert.
The military has seized power in all three countries, and formed the Alliance of Sahel States.
They have cut ties with former colonial power France and befriended Russia instead, buying weapons and deploying fighters with the mercenary Wagner Group, now known as the Africa Corps.
However, armed groups have stepped up their attacks, particularly in Burkina Faso, despite massive recruitment by the paramilitary Volunteers for the Defence of the Homeland, a self-defence militia.
'''

QA = Generate_question(Article, 3, num_generations=1)