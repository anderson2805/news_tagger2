from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import spacy
from flashtext import KeywordProcessor
import os
import utils
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
            entity = []
            for index, token in enumerate(spacy_nlp(ent.text)):
                if not ((index == 0 and token.is_stop) or token.is_punct):
                    if (token.is_stop):
                        entity.append(token.text)
                    else:
                        entity.append(token.text.title())

            ner_kw.append(        
            " ".join(entity))
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
    doc = """TODAY Online (28 May) carried a forum letter by Ho Hua Chew in reference to a report published on 26 May (2020) titled "Researchers call for protection of sf training area to preserve feeding ground for Raffles' banded langur". The writer opined that at present, the forest to the north of Tagore Drive and the Tagore industrial estate was being used by the sf for training
it had become even more important to wildlife in the Upper Thomson area, given that the forest patch to the south of it fringing Yio Chu Kang Road and the Teachers’ Estate — which was called the Tagore-Lentor Forest — was almost completely wiped out for a condominium development
from as early as 2001, Singapore also had the critically endangered songbird, the straw-headed bulbul, in the forest patch north of Tagore Drive the straw-headed bulbul was listed as critically endangered in the International Union for Conservation of Nature’s (IUCN) Red List of Threatened Species
given the demise of the neighbouring, connected forest around the Yio Chu Kang and Teachers’ Estate fringe, where there were also records of this bulbul species, it was most probable that the bulbuls here would take refuge in the forest north of Tagore Drive through a narrow forest belt to the east of the Tagore industrial estate
there were also records of other nationally threatened bird species, such as the crested serpent eagle and the grey-headed fish eagle, in this patch north of Tagore Drive the grey-headed fish eagle was also in the IUCN’s Red List as “near-threatened”
The Nature Society (Singapore) also believed that the Sunda pangolin, another critically endangered species globally, would have likewise taken refuge in this patch north of Tagore Drive the pangolin had been recorded in the forest patch fringing the Teachers’ Estate, which was, as mentioned, already mostly cleared
with the presence of the Raffles’ banded langur, as primatologist Andie Ang noted in the TODAY Online report, the forest north of Tagore Drive was highly important for the well-being of Singapore's biodiversity and the Nature Society (Singapore) urged the authorities to do an environmental or a biophysical impact assessment, to determine at least some ecologically significant portion of the forested area for conservation before initiating any housing plan.


(The writer is Vice-President of the Nature Society (Singapore).)"""

    print(usertags_exist)
    print(get_tags(doc))
