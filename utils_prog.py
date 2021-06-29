import re
import sys


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


stop_words = {
    'be', 'other', 'thru', 'hers', 'one', 'another', 'nothing', 'those', 'the',
    'when', 'where', 'if', 'thereupon', 'so', 'really', 'should', 'a', 'an', 'on', 'of',
    'give', 'whom', 'neither', 'could', 'at', 'once', 'doing', 'whereafter', 'even', 'has',
    'its', 're', 'whether', 'since', 'empty', 'too', 'his', 'last', 'perhaps', 'that', 'off'
    'ca', 'in', 'now', 'over', 'thereafter', 'front', 'serious', 'else', 'for', 'as', 'then',
    'latterly', 'whereas', 'did', 'itself', 'everything', 'though', 'herself', 'or',
    'each', 'forty', 'any', 'have', 'such', 'around', 'towards', 'nowhere', 'after', 
    'toward', 'latter', 'which', 'yourself', 'with', 'four', 'become', 'own',
    'themselves', 'whereby', 'throughout', 'thus', 'by', 'been', 'only', 'and',
    'nor', 'myself', 'whoever', 'yourselves', 'their', 'cannot', 'beyond',
    'to', 'two', 'bottom', 'can', 'than', 'made', 'below', 'hereby', 'keep',
    'various', 'enough', 'anything', 'also', 'done', 'alone', 'somehow',
    'them', 'him', 'upon', 'we', 'what', 'twenty', 'please', 'indeed',
    'under', 'still', 'elsewhere', 'others', 'were', 'whereupon', 'why', 'between', 'here', 'had', 
    'behind', 'within', 'they', 'because', 'anyhow', 'quite', 'call', 'being', 'off', 'however', 
    'everyone', 'get', 'formerly', 'is', 'former', 'himself', 'ourselves', 'it', 'he',
    'this', 'but', 'are', 'who', 'not', 'from', 'you', 'was', 'her', 'all', 'out', 'there', 'about', 'up',
    'she', 'some', 'into', 'will', 'much', 'would', 'just', 'through', 'me', 'do', 'does', 'my', 'us',
    'film','films', 'movie', 'movies'}


def clean_text_for_comparison(input: str, remove_stop_words: bool=True):
    '''
    This is to clean the header text to compare with dictionary items.
    remove everything other than alphabets.
    '''
    regex = re.compile('[^a-zA-Z]')
    result = regex.sub(' ', input).lstrip().rstrip().lower()
    if(remove_stop_words):
        result = ' '.join(
            [r for r in result.split()
             if not stop_words.__contains__(r) and len(r) != 1])

    return result
