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
    ner_kw = list(set(ner_kw) - set(existing_kw))

    ner_kw = [keyword[0]
              for keyword in bert_nlp.extract_keywords(docs=doc, candidates=ner_kw)]
    topic_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(
        docs=doc, candidates=candidate_kw)]
    theme_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(
        docs=doc, vectorizer=vectorizer, use_mmr=True) if len(keyword[0].split(' ')) <= 5 ]
    combined_kw = [keyword[0] for keyword in bert_nlp.extract_keywords(docs=doc, candidates=list(
        set(existing_kw+ner_kw+topic_kw+theme_kw)), use_mmr=True, top_n=top_n)]
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
    doc = """A commentary was written by Rajan Menon on the long war of attrition in Ukraine.
Despite the Ukrainian army's battlefield advances and Russia's retreats, most recently from parts of Kherson province, Ukraine's economy had been left in tatters.
For the Kyiv government, the cost of prosecuting the war while also meeting the material needs of its citizens would mount even if the Ukrainian army kept gaining ground.
Worse, winter was looming and Russia, frustrated by the serial military failures it had experienced since Sep, seemed bent on crippling Ukraine's economy by taking the wrecking ball to its critical infrastructure.
Aid to Ukraine would not dry up, nor would the Ukrainian economy collapse, but Western governments might find it harder, politically if not economically, to keep sending billions of dollars to Kyiv while their own citizens endured rising prices and increasing joblessness.
Poland, Germany, and Hungary were now struggling to accommodate more Ukrainian refugees, and the mood in Europe had become less welcoming just when the outflow from Ukraine had picked up, following Russia's ramped-up attacks on cities.
    """
    print(usertags_exist)
    print(get_tags(doc))
