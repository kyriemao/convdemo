import time
from clarification.codes.config import *
from clarification.codes.utils import *
from clarification.codes.bart_model import *
from clarification.codes.policy import *
from IPython import embed

def clarify(query, bartres, srqg, intent_cqgm, RETRIEVAL, SUGGESTION, CLARIFICATION, INFORMATIVE, INTENT_VERB):
    print('system initialization finished!')

    query = query.lower()
    search_result_text = ''

    # -------------------- retrieval module --------------------
    if RETRIEVAL:
        start = time.time()
        get_result(query, CACHE_JSON_PATH + query + '.json')
        concat_query, titles, urls, snippets = parse_result(CACHE_JSON_PATH + query + '.json')
        end = time.time()

        print('\nFound ' + str(len(titles)) + ' search results (' + str(end - start) + ' seconds)\n')
        for i in range(min(10, len(titles))):
            print(i + 1, titles[i], urls[i])

        search_result_text = ' [SEP] '.join(snippets)

    suggestions = get_suggestion(query)  # get query suggestions

    # -------------------- query suggestion module --------------------
    if SUGGESTION:
        print('\nquery suggestion:')
        for i in range(len(suggestions)):
            print(suggestions[i])

    # -------------------- clarification module --------------------
    clarifying_question = ''
    aspect_items = []
    if CLARIFICATION:
        beam_result = bartres.predict(concat_query, num_beams=20)
        clarifying_question, aspect_items = clarification_post_process(query, beam_result, suggestions,
                                                                       search_result_text)

        print('\nclarification pane:')
        print('original clarifying question: ' + str(clarifying_question).capitalize())
        with open(CACHE_PATH + query + '.txt', 'a', encoding='utf-8') as f:
            f.write('original clarifying question: ' + str(clarifying_question).capitalize() + '\n')

        # -------------------- informative clarification module --------------------
        informative_question = ''
        if INFORMATIVE:
            if clarifying_question in ['select one to refine your search',
                                       'what do you want to know about ' + query + '?',
                                       'what would you like to know about ' + query + '?']:
                beam_result = srqg.predict(query + ' [SEP] ' + ' [ISEP] '.join(aspect_items))
                beam_result = informative_question_post_process(beam_result, query)
                informative_question = beam_result[0]
                if informative_question.strip() != clarifying_question.strip():
                    print('informative clarifying question: ' + str(informative_question).capitalize())
                    with open(CACHE_PATH + query + '.txt', 'a', encoding='utf-8') as f:
                        f.write('informative clarifying question: ' + str(informative_question).capitalize() + '\n')

        # -------------------- intent-aware clarification module --------------------
        intent_question = ''
        if INTENT_VERB:
            if informative_question != '':
                if informative_question == 'select one to refine your search':
                    informative_question = 'what do you want to know about ' + query + '?'
                informative_question = informative_question.replace('would you like', 'do you want')
                beam_result = intent_cqgm.predict(query + ' [SEP] ' + ' [ISEP] '.join(aspect_items) + ' [QSEP] ' +
                                                  informative_question)

                intent_question = beam_result[0]
                if intent_question.strip() != informative_question.strip():
                    print('intent-aware clarifying question: ' + str(intent_question).capitalize())
                    with open(CACHE_PATH + query + '.txt', 'a', encoding='utf-8') as f:
                        f.write('intent-aware clarifying question: ' + str(intent_question).capitalize() + '\n')
            else:
                if clarifying_question == 'select one to refine your search':
                    clarifying_question = 'what do you want to know about ' + query + '?'
                clarifying_question = clarifying_question.replace('would you like', 'do you want')
                beam_result = intent_cqgm.predict(query + ' [SEP] ' + ' [ISEP] '.join(aspect_items) + ' [QSEP] ' +
                                                  clarifying_question)

                intent_question = beam_result[0]
                if intent_question.strip() != clarifying_question.strip():
                    print('intent-aware clarifying question: ' + str(intent_question).capitalize())
                    with open(CACHE_PATH + query + '.txt', 'a', encoding='utf-8') as f:
                        f.write('intent-aware clarifying question: ' + str(intent_question).capitalize() + '\n')

        print('aspect items: ' + ', '.join(aspect_items))
    print('\n---------------------------------------------------------------------------------------\n')

    return clarifying_question, informative_question, intent_question, aspect_items


def load_models():
    bartres = BART(BARTRES_PATH, 64)
    srqg = BART(SRQG_PATH, 32)
    intent_cqgm = BART(CQGM_PATH, 32)
    return bartres, srqg, intent_cqgm


# # 1. load necessary models
# bartres, srqg, intent_cqgm = load_models()

# # 2. run 'clarify' function

# print("please input:")
# query = input()
# x = clarify(query=query,
#                bartres=bartres,
#                srqg=srqg,
#                intent_cqgm=intent_cqgm,
#                RETRIEVAL=True,
#                SUGGESTION=True,
#                CLARIFICATION=True,
#                INFORMATIVE=True,
#                INTENT_VERB=True)
# embed()
# input()