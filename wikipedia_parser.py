from urllib.parse import urlparse
from bs4 import BeautifulSoup
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

import requests
import re
import numpy as np
import csv


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = []
  
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)      
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence)) 
                                       if noun not in self.stopwords and len(noun) > 1]))
        
        return nouns



class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return  self.graph_sentence
        
    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}



class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}



class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
                    
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    def summarize(self, sent_num=3):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        
        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
            
        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        
        return keywords



class parser_pack:
    def html_to_bsObj(url):
        req = requests.get(url)
        html = req.text
        bsObj = BeautifulSoup(html,"html.parser")

        return bsObj

    def document_url_distinguish(wiki_document):
        bs4_document_url = wiki_document.bsObj.find("div",{"class":{"mw-parser-output"}}).findAll("a",href=re.compile("^(/wiki/)"))

        for link in bs4_document_url:
            if link.attrs['href'] is not wiki_document.document_url:
                wiki_document.document_url.append(link.attrs['href'])

        bs4_outside_url = wiki_document.bsObj.find("div",{"class":{"mw-parser-output"}}).findAll("a",href=re.compile("^(http://|https://)"))
        for link in bs4_outside_url:
            if link.attrs['href'] is not wiki_document.outside_url:
                wiki_document.outside_url.append(link.attrs['href'])

    def repacking_doc(wiki_document):
       # bs4_document_thumbs =
       # wiki_document.bsObj.find("div",{"class":{"mw-parser-output"}}).findAll("div",{"class":"thumb
       # tright"})
        bs4_document_thumbs = wiki_document.bsObj.find("div",{"class":{"mw-parser-output"}}).findAll("div")
        bs4_document_tables = wiki_document.bsObj.find("div",{"class":{"mw-parser-output"}}).findAll("table")
        temp_html = str(wiki_document.bsObj)
        
        for bs4_table in bs4_document_tables:
            temp_html = temp_html.replace(str(bs4_table),"")
        
        for bs4_thumb in bs4_document_thumbs:
            temp_html = temp_html.replace(str(bs4_thumb),"")

        bsObj = BeautifulSoup(temp_html,"html.parser")
        bs4_packed = bsObj.find("div",{"class":{"mw-parser-output"}})
        temp_html = str(bs4_packed)

        wiki_document.bsObj = BeautifulSoup(temp_html,"html.parser")

    def delete_ref(wiki_document):
        p1 = re.compile("\[[^\]]*\]")
        p2 = re.compile("\([^\)]*\)+")
        wiki_document.doc_main = p1.sub("",wiki_document.doc_main)
        wiki_document.doc_main = p2.sub("",wiki_document.doc_main)




class wiki_document:
    def __init__(self, document_url):
        self.document_url = []
        self.outside_url = []
        self.doc_sector = [] #쓰이지 않음
        self.bsObj = parser_pack.html_to_bsObj("https://ko.wikipedia.org" + document_url)
        self.title = self.bsObj.find("h1",{"class":"firstHeading"})
        self.title = self.title.get_text()

        parser_pack.document_url_distinguish(self)
        parser_pack.repacking_doc(self)
        self.doc_main = self.bsObj.get_text()
        parser_pack.delete_ref(self)


if __name__ == "__main__":
    doc = wiki_document('/wiki/' + input("위키 문서 제목: "))
    tr = TextRank(doc.doc_main)
    for row in tr.summarize(10):
        print(row)
        print()

    print('keywords :',tr.keywords(30))

'''
if __name__=="__main__":
    
    doc = wiki_document('/wiki/컴퓨터')
    tr = TextRank(doc.doc_main)
    print('---------------------공학에 대한 단어 검색-----------------------')
    words_tf_idf_line = {}
    for name, age in tr.idx2word.items():
        if age == "공학":
            for row in range(0,len(tr.idx2word)):
                words_tf_idf_line[tr.idx2word[row]] = tr.words_graph[row][name]
            
            for y,v in sorted(words_tf_idf_line.items(), key = lambda words_tf_idf_line:words_tf_idf_line[1]):
                print(y,v)

            break;
 
    
 
    f = open('test.csv', 'w', encoding='ANSI', newline='')
    wr = csv.writer(f)
        
    wr.writerow([doc.title])
    wr.writerow(doc.document_url)
    
    words_list  = ['TF-IDF']
    for i in range(0,len(tr.idx2word)):
        words_list.append(tr.idx2word[i])

    wr.writerow(words_list)
    for i in range(0,len(tr.idx2word)):
        wr.writerow([tr.idx2word[i]]+words_tf_idf[i].tolist())


    f.close()    
 '''