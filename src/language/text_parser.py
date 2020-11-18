from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector
import string

class EkphrasisParser:
    def __init__(self):
        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number', 'hashtag'],
            #normalize=['url', 'email', 'percent', 'money', 'phone', 'time', 'url', 'date', 'number'],
            # terms that will be annotated
            #annotate={"hashtag", "allcaps", "elongated", "repeated",'emphasis', 'censored'},
            annotate={"allcaps", "elongated", 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter="twitter", 

            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector="twitter", 

            unpack_hashtags=False,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )
    
    def parseText(self, text):
        #print(text)
        tmp_tokens = self.text_processor.pre_process_doc(text)
        #print(tmp_tokens)
        processed_tokens = [wd for wd in tmp_tokens if wd != 'rt' and wd != '<user>' and wd != '<url>' 
                            and wd != '<censored>' and wd != '<elongated>' and wd != '<hashtag>' and wd != '<allcaps>' 
                            and wd != '</censored>' and wd != '</elongated>' and wd != '</hashtag>' and wd != '</allcaps>']
        #print(processed_tokens)
        processed_text = " ".join(processed_tokens)
        #print(processed_text)
        final_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        return " ".join(final_text.split())
        
        #cat, dic = liwc.read_liwc('liwc_data/LIWC2007_English131104.dic')
        #result = liwc.getLIWCFeatures(cat, dic, processed_sentence)
        #print(result)

        #print(liwc.get_text2word(cat, dic, processed_sentence))
        #print(liwc.get_text2cat(cat, dic, processed_sentence))

