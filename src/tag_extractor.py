from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import spacy
from collections import Counter
from flashtext import KeywordProcessor
import os
from collections import defaultdict
USERTAGSPATH = "data/usertags.txt"
usertags_exist = os.path.isfile(USERTAGSPATH)


BERTPATH = "./models/all-MiniLM-L6-v2"
SPACYPATH = "./models/en_core_web_md"

vectorizer = KeyphraseCountVectorizer(
    spacy_pipeline=SPACYPATH, lowercase=False)

spacy_nlp = spacy.load(SPACYPATH)
bert_nlp = KeyBERT(BERTPATH)
keyword_processor = KeywordProcessor(case_sensitive=False)


# TODO: For production, generate usertags.txt before running tags extraction on articles
keyword_processor.add_keyword_from_file(USERTAGSPATH)
usertags_list = open(USERTAGSPATH, "r").read().split("\n")


def get_tags(doc: str, usertags_exist: bool = usertags_exist, top_n: int = 10) -> dict:
    """
    Suggest up to 15 tags based on the title and summary of articles.
    3 types of keywords are suggested:
    -Entity: By direct search on doc for existing tags list (using keyword_processor) and Entities from Spacy NER, all the keywords are combined to find top 5 related to doc.
    -Topic: By zero shot classifier, finding top 5 related to candidate keys (which are user tags not matched on direct search)
    -Keyphrases(theme): By using KeyphraseCountVectorizer, return top 5 phrases that best represent the doc

    Args:
        doc (str): title and summary of article in single string variable
        usertags (bool): True if usertags.txt exist

    Returns:
        dict: return dict of keywords with 3 keys, user_tags, NER, keyphrase, topics (in no particular order) 
    """

    if usertags_exist == True:
        existing_kw = list(set(keyword_processor.extract_keywords(doc)))
    else:
        existing_kw = []

    ner_kw = []
    ner_nlp = spacy_nlp(doc)
    for ent in ner_nlp.ents:
        if (ent.label_ in ['GPE', 'PERSON', 'ORG', 'LOC']):
            ner_kw.append(
                " ".join([token.text for token in spacy_nlp(ent.text) if not token.is_stop]))
    ner_kw = list(set(ner_kw))
    # Delta on user tags not identified, candidate_kw is used for zero-shot
    candidate_kw = list(set(usertags_list) - set(existing_kw))
    print(set(existing_kw))
    ner_kw = list(set(ner_kw) - set(existing_kw))

    ner_kw = [keyword[0]
              for keyword in bert_nlp.extract_keywords(docs=doc, candidates=[x.lower() for x in ner_kw])]
    topic_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(
        docs=doc, candidates=[x.lower() for x in candidate_kw])]
    theme_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(
        docs=doc, vectorizer=vectorizer, use_mmr=True) if len(keyword[0].split(' ')) <= 5 ]
    combined_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(docs=doc, candidates=list(
        set([x.lower() for x in existing_kw]+ner_kw+topic_kw+[x.lower() for x in theme_kw])), use_mmr=True, top_n=top_n)]
    # combined all keywords extracted
    #result_kw = list(set([keyword[0] for keyword in topic_kw] + [keyword[0] for keyword in entity_kw] + [keyword[0] for keyword in theme_kw]))
    results = {}
    results['ExistingTags'] = existing_kw
    results['NERTags'] = ner_kw
    results['TopicTags'] = topic_kw
    results['ThemeTags'] = theme_kw
    results['TopNTags'] = [" ".join([word[0].upper() + word[1:] for word in words.split(' ')]) for words in combined_kw]

    return results


if __name__ == '__main__':
    doc = """PM Albanese Says He’s Not In Group Of Aussie Politicians Visiting Taiwan. Australia's Prime Minister Anthony Albanese said on 3 Dec (2022) that he would not be part of a group of federal politicians set to travel to Taiwan for a reported five-day visit aimed at conveying Canberra's wish to maintain peace in the Indo-Pacific. The report noted that according to Australian, the group, which included Australia's governing Labor Party and opposition Liberal-National coalition MPs, would fly to Taiwan on 4 Dec (2022) and would be the first delegation of its type to visit Taiwan since 2019 Mr Albanese had described the trip as a “backbench” visit to Taiwan, not a government-led one Mr Albanese said “There remains a bipartisan position when it comes to China, and when it comes to support for the status quo on Taiwan”
when asked about the travelling politicians' intentions, Mr Albanese said “I have no idea, I'm not going, you should ask them” an Australian Department of Foreign Affairs and Trade spokesperson said politicians from various parties regularly travelled to Taiwan before the COVID-19 pandemic and that the current delegation “represents a resumption of that activity” the group would reportedly meet Taiwan President Tsai Ing-wen and Foreign Minister Joseph Wu, with the visit having support from Taiwan's Foreign Ministry the trip – reportedly kept secret to stop Chinese diplomats in Canberra lobbying for its cancellation – was said to include meetings on security, trade, agriculture and indigenous affairs the visit to Taiwan came as Australia's recently elected Labor government had moved to repair its strained diplomatic relations with China and Australia, like most countries, had no official diplomatic ties with Taiwan, but had previously joined the US in expressing concern over Chinese pressure, especially in military issues. 
    """
    print(usertags_exist)
    print(get_tags(doc))
