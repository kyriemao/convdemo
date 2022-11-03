import os
import json
import functools
import requests
import http.client
import urllib.parse
import numpy as np
from clarification.codes.config import *


def get_result(query, file_name):
    # get the json format search result of 'query'
    if os.path.exists(file_name):  # if the file_name already exists, return directly
        print(file_name, 'already exists')
        return

    headers = {"Ocp-Apim-Subscription-Key": BING_RESULT_KEY, "Accept-Language": "en-US"}
    params = {"q": query, "textDecorations": True, "textFormat": "HTML", "count": 100, "cc": "US"}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    search_result = response.json()

    with open(file_name, 'w', encoding='utf-8') as fp:
        fp.write(str(search_result))
    reformat_json(file_name)
    print('results of query "' + query + '"' + ' have been downloaded into file ' + file_name)


def reformat_json(file_name):
    # convert single quotes to double quotes, convert True to "True", False to "False", etc. to support json parsing
    with open(file_name, 'r', encoding='utf-8') as fp:
        content = fp.read()
        # content = content.replace("'", '"')
        content = content.replace("': True", "': True").replace("': False", "': False").replace("': null", "': None")
        content = content.replace('\\x', ' ').replace('<b>', '').replace('</b>', '')
    with open(file_name, 'w', encoding='utf-8') as fp:
        fp.write(content)


def parse_result(file_name):
    # parse a json file, return result_dict, query, title_list, url_list
    try:
        with open(file_name, 'r', encoding='utf-8') as fp:
            search_result = fp.read()

        json_dict = eval(search_result)  # convert to dict
        # get query, title list, url list
        query = json_dict['queryContext']['originalQuery']
        title_list = []
        url_list = []
        snippet_list = []
        for page in json_dict['webPages']['value']:
            try:
                title_list.append(page['name'])
                url_list.append(page['url'])
                snippet_list.append((page['name'] + ' # ' + page['snippet']).lower().strip())
            except Exception:
                continue

        concat_query = query + ' [SEP] ' + ' [SEP] '.join(snippet_list)
        return concat_query, title_list, url_list, snippet_list
    except Exception:
        return 'error', 'error', 'error', 'error'


def get_suggestion(query):
    original_query = query
    query = '+'.join(query.split(' ')) + '+'
    subscriptionKey = BING_SUGGESTION_KEY
    host = 'api.bing.microsoft.com'
    path = '/v7.0/Suggestions'
    mkt = 'en-US'
    params = '?mkt=' + mkt + '&q=' + query
    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    try:
        conn = http.client.HTTPSConnection(host)
        conn.request("GET", path + params, None, headers)
        response = conn.getresponse().read()
        response = json.loads(response)
        suggestions = []
        for i in range(len(response['suggestionGroups'][0]['searchSuggestions'])):
            suggestions.append(response['suggestionGroups'][0]['searchSuggestions'][i]['query'])
        return suggestions
    except Exception:
        print('error when getting query suggestions')
        return []


def clarification_post_process(query, beam_results, suggestions, search_result_text):
    question_set, items_set = [], []
    query_log_scores = []
    question_template_scores = []
    length_scores = []
    generate_frequency_scores = []
    # 1. query log, 2. question_template, 3. length, 4. generate_frequency
    generate_frequency_dict = {}

    for i in range(len(beam_results)):
        try:
            question, items = beam_results[i].split(' [QSEP] ')
            question_set.append(question)
            items_set.append(items.split(' [ISEP] '))
        except Exception:
            continue

    for i in range(len(question_set)):
        log_score = 0
        for item in items_set[i]:
            if item in ' '.join(suggestions):
                log_score += 1

            if item not in generate_frequency_dict.keys():
                generate_frequency_dict[item] = 1
            else:
                generate_frequency_dict[item] += 1

        query_log_scores.append(log_score)
        length_scores.append(len(items_set[i]))

        if question_set[i] == 'select one to refine your search' or \
                ('what do you want' in question_set[i] and
                 ' this ' not in question_set[i] and ' the ' not in question_set[i]) or \
                ('what would you like' in question_set[i] and
                 ' this ' not in question_set[i] and ' the ' not in question_set[i]):
            question_template_scores.append(0.0)
        else:
            question_template_scores.append(1.0)

    for i in range(len(items_set)):
        generate_frequency_scores.append(
            np.average([generate_frequency_dict[item] for item in items_set[i]]))

    def cmp(this, other):
        if query_log_scores[this] > query_log_scores[other]:
            return -1
        elif query_log_scores[this] < query_log_scores[other]:
            return 1
        else:
            if question_template_scores[this] > question_template_scores[other]:
                return -1
            elif question_template_scores[this] < question_template_scores[other]:
                return 1
            else:
                if length_scores[this] > length_scores[other]:
                    return -1
                elif length_scores[this] < length_scores[other]:
                    return 1
                else:
                    if generate_frequency_scores[this] > generate_frequency_scores[other]:
                        return -1
                    elif generate_frequency_scores[this] < generate_frequency_scores[other]:
                        return 1
                    else:
                        return 0

    indexes = range(len(question_set))
    indexes = sorted(indexes, key=functools.cmp_to_key(cmp))

    with open(CACHE_PATH + query + '.txt', 'w', encoding='utf-8') as f:
        f.write('Question\tItems\tQuery log score\tTemplate score\tLength score\tGenerate frequency score\n')
        for i in range(len(question_set)):
            f.write(question_set[i] + '\t' + str(items_set[i]) + '\t' + str(query_log_scores[i]) + '\t' + str(
                question_template_scores[i]) + '\t' + str(length_scores[i]) + '\t' + str(
                generate_frequency_scores[i]) + '\n')

    best_question = question_set[indexes[0]]
    raw_items = items_set[indexes[0]]
    best_items, expanded_items = items_expansion(raw_items, items_set, search_result_text)
    with open(CACHE_PATH + query + '.txt', 'a', encoding='utf-8') as f:
        f.write('--------------------------------------------------------------------------------\n')
        f.write('suggestions: ' + str(suggestions) + '\n')
        f.write('seed items: ' + str(raw_items) + '\n')
        f.write('final items: ' + str(best_items) + '\n')

    return best_question, best_items


def read_pre_defined_items_set():
    with open(PRE_DEFINED_ITEMS_SET, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')[:-1]
    return [eval(line) for line in lines]


def items_expansion(seed_items, items_set, search_result_text):
    # obtain co-occured items
    new_items = []
    for i in range(len(items_set)):
        for j in range(len(items_set[i])):
            if items_set[i][j] in seed_items:
                new_items.extend(items_set[i])
                break

    expanded_items = [seed for seed in seed_items]

    for item in new_items:
        item = item.replace('#hash#', '#')
        if search_result_text.find(item) != -1:
            expanded_items.append(item)
    best_items = list(set(expanded_items))

    # supplement pre-defined items
    pre_defined_items_set = read_pre_defined_items_set()
    for i in range(len(pre_defined_items_set)):
        pre_defined = pre_defined_items_set[i]
        count = 0
        for item in best_items:
            if item in pre_defined:
                count += 1
        if count == len(best_items):
            best_items = pre_defined

    if len(best_items) <= 5:
        return best_items, expanded_items
    else:  # return best five items
        items_freq_dict = {}
        for item in best_items:
            items_freq_dict[item] = search_result_text.count(item)
        sorted_dict = sorted(items_freq_dict.items(), key=lambda x: x[1], reverse=True)
        best_items = []
        for i in range(5):
            best_items.append(sorted_dict[i][0])
        return best_items, expanded_items


def informative_question_post_process(beam_results, query):
    ban_words = ['select one', 'story element', 'risk factor', 'what would you like to know about this ' + query + '?',
                 'what do you want to know about this ' + query + '?', 'which window are you looking for?',
                 'what would you like to know about ' + query + '?', 'what do you want to know about ' + query + '?']
    new_results = []
    for i in range(len(beam_results)):
        ok = 1
        for w in ban_words:
            if w in beam_results[i]:
                ok = 0
        if ok:
            new_results.append(beam_results[i])
    return new_results


def clarification_policy(aspect_items):
    return True
