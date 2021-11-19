#%% KÜTÜPHANELER
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from tensorflow.python.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU
from tensorflow.keras.optimizers import Adam
import kok_bulma as kb
import make_regex 

from nltk.corpus import stopwords

import re
import warnings
warnings.filterwarnings("ignore")

#%% VERİSETİNİ OKUMA..

dataset=pd.read_csv("dataset1.csv")
dataset.drop(["Unnamed: 0"],axis=1,inplace=True)

dataMail=dataset.iloc[:,0]  #Bağımsız değişkenimiz
target=dataset.iloc[:,1]    #Bağımlı değişkenimiz...

#Verileri listeye çevirme..
dataMail=dataMail.values.tolist()
target=target.values.tolist()

dataMail[1]
target[1]

#%%Verileri "metni_isle" adında fonksiyona sokalım.

sentences=[]
for i in range(len(dataMail)):
    a=make_regex.metni_isle(dataMail[i])
    sentences.append(a)
    
len(dataMail)

dataMail[4]
sentences[4]

#%% Harf harf ayıralım...

x=[]
for i in sentences:
    x.append(i.split())
    
y=[]
for i in range(len(x)):
    for j in range(len(x[i])):
        y.append(x[i][j])

z=[]
for k in range(len(y)):
    z.append(list(y[k]))


#%%Köklerine ayırma
    
"split_stem fonksyionu oluşturalım"
def split_stem(take_sentences):
    
    newList=[]
    for i in take_sentences:
        newList.append(list(i))
    
    
    for i in range(len(newList)):
        for j in newList[i]:
            if j == '̇':
                newList[i].remove(j)
    
    asd=[]
    for i in range(len(newList)):
        asd.append(''.join(newList[i]))
        
    newSentences = kb.kok_bul(asd)
    
    
    newSentencesfinally=[]
    for i in range(len(newSentences)):
        b = " ".join(newSentences[i])
        newSentencesfinally.append(b)
    
    return newSentencesfinally
    
    
#%% TOKENLEŞTİRME VE WORD2VEC MODELİ OLUŞTURMA...

tokenizer = Tokenizer()
stem_token = split_stem(sentences)
tokenizer.fit_on_texts(stem_token)
from itertools import islice
def take(n,iterable):  
    return islice(iterable,n)
    
type(tokenizer.word_index.items())  

tokenizer.word_index.items() # 3129 tane kelimemiz var totalde
n_items=take(420,tokenizer.word_index.items()) # En cok gecen 420 kelimeyi aldım

print(str(dict(n_items)))

#Word2Vec modeli...
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

corpus=[]
for w in sentences:
    corpus.append(w.split()) #split ile kelime kelime ayırdım ki kelimeler arasındaki ilişkileri verebilsin...    
    
print(corpus[:5])
type(corpus)

model = Word2Vec(corpus, size=100, window=5, min_count=5, sg=1)#Skip-Gram ile oluşturuldu...

model.wv['fiyat']
model.wv.most_similar('fiyat')

#Word2Vec Modelini Grafiğe Dökme...
def closestwords_tsneplot(model, word):
    word_vectors = np.empty((0,100))
    word_labels = [word]
    
    close_words = model.wv.most_similar(word)
    
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)
    
    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis=0)
        
    tsne = TSNE(random_state=0)
    Y = tsne.fit_transform(word_vectors)
    
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    plt.scatter(x_coords, y_coords)
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords='offset points')
        
    plt.show()

closestwords_tsneplot(model, 'fiyat')

#%% WORD CLOUD OLUŞTURMA
    
# text=sentences[0]
# sentences[0]

# from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
# wordcloud=WordCloud(max_font_size=50,max_words=100,background_color="black").generate(text)   
# plt.imshow(wordcloud,interpolation="bilinear")
# plt.axis("off")
# plt.show()

# #datasetteki tüm mailleri alalım
# text=" ".join(i for i in sentences)
# wordcloud=WordCloud(max_font_size=50,background_color="black").generate(text)
# plt.figure(figsize=[10,10])
# plt.imshow(wordcloud,interpolation="bilinear")
# plt.axis("off")
# plt.show()

# #Png Olarak Kaydetme...
# wordcloud.to_file("kelime_bulutu.png") 

#%% Pickle Oluşturma

import pickle
with open("tokenizer.pickle","wb") as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)    
    

#%% MODEL OLUŞTURMA VE EĞİTME

dataset=pd.read_csv("datasetvideo.csv")
dataset.drop(["Unnamed: 0"],axis=1,inplace=True)

dataMailvideo=dataset.iloc[:,0]
targetvideo=dataset.iloc[:,1]

dataMailvideo=dataMailvideo.values.tolist()
targetvideo=targetvideo.values.tolist()
    
#%%Verileri "metni_isle" adında fonksiyona tekrardan sokalım.

def metni_isle(metin):
    #mersis kelimelerini silelim..
    sablon=r'\bmersis:|mk:|m:|mrs:\b'
    regex=re.compile(sablon, re.I)
    metin = regex.sub(' ', metin)
    
    #mersis numaralarını silelim..
    sablon=r'\b\d{16}|\d{15}\b'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' MersisNo ', metin)
    
    # operatör numaralarını silelim...
    sablon = r"\bB[0-9]{3}\b"
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' ', metin)

    # fiyatları silelim...
    sablon = r'(\d{1,3}((,|\.)\d{3})*|\d+)((,|\.)\d*)?(\s*tl)'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' Fiyat', metin)
    
    # yüzdeleri, oranları silelim...
    sablon = r'(\d{1,3})((,|\.)\d{1,2})?(\s?%)|(%\s?)(\d{1,3})((,|\.)\d{1,2})?'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' Yüzde ', metin)

    # URL'leri silelim...
    sablon = r'((([A-Za-z]{3,9}(:\/\/))(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.-]+)((?:\/[\+~%\/.\w_]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[.\!\/\\w]*))?)'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' URL ', metin)
    
    # zamanı silelim
    sablon = r'([0-9]|0[0-9]|1[0-9]|2[0-3])[:.][0-5][0-9]'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' ', metin)
    
    #tarihi silelim...
    sablon=r'((([1-9])|(0[1-9])|([12])([0-9]?)|(3[01]?))(-|\/|\.)(0?[13578]|10|12)(-|\/|\.)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1}))|(([1-9])|(0[1-9])|([12])([0-9]?)|(3[0]?))(-|\/|\.)(0?[2469]|11)(-|\/|\.)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1})))'
    regex = re.compile(sablon, re.I)
    metin = regex.sub('Tarih', metin)


    # telefon numaralarını silelim..
    sablon = r'(?:\b\d{11}\b|\b\d{12}\b)'
    regex = re.compile(sablon, re.I)
    metin = regex.sub(' TelefonNo ', metin)   
                                                
                                                                                             
    # alfa nümeric sayıları silelim...
    metin = re.sub('[^\w]', ' ', metin)
    
    # türkçe karaktere çevirelim...
    trCharList = str.maketrans("E", "e")
    metin = metin.translate(trCharList)


    #bütün kelimeleri küçük harflere çevirelim...
    metin=metin.lower()
    
    # mailleri tokenlestirelim...
    kelimeler = word_tokenize(metin)


    # Stopwords'leri silelim...
    stopword_list = stopwords.words('turkish')
    # türkçe karakterleri değiştirelim...
    for i in range(len(stopword_list)):
         stopword_list[i] = stopword_list[i].translate(trCharList)


    # yeni stopwords'ler ekleyelim...
    stopword_list = stopword_list + ['li', 'lu', 'lik', 'e', 'ye', 'ini',
    'i', 'a', 'com', 'tr', 'org', 'net', 'den',
    'no', 'tan','larda','Lerde','yi','nda','nde','u','ları','leri','ebilir','abilir'
    'nda','nde','icin','ta','te']
    
    kelimeler = [w for w in kelimeler if w not in set(stopword_list)]
    return  ' '.join(kelimeler)


#%% VERİLERİ SPLİT ETME VE TOKENLESTİRME ADIMI
    
sentencesvideo=[]
for i in range(len(dataMailvideo)):
    a=metni_isle(dataMailvideo[i])
    sentencesvideo.append(a)

len(sentencesvideo)

#%%
#TRAİN TEST SPLİT

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sentences,target,test_size=0.20,random_state=0)

len(x_train)
len(x_test)

#Orjinal verisetimizdeki mail
dataMail[4]
    
#Veri önişlemeye girmiş mail
# metni_isle("Tüm Tabletlerde KDV tutarı kadar indirim ve SAMSUNG 49NU7100 124 Ekran 4K UHD TV 3597TL yerine 2998TL.Son gün: 2 Eylül.Stoklarla sınırlı fırsatları kaçırmayın.Detaylar:Teknosa Mağazaları'nda ve https://bit.ly/2HwYv9k .SMS ret için TSA RET yazıp 7889'a ücretsiz gönderebilirsiniz.")

x_train[4]
y_train[4]

import json
with open('last_token.json') as json_dosyasi: #Daha önceden oluşturdugum "tokenizer.json" dosyasını "json_tokenizer" olarak yüklüyorum.
    json_tokenizer=json.load(json_dosyasi)
    
# json_tokenizer[""]

# "tokenlestir" fonksiyonu
def make_token(mails):   #Padding fonksyionu kullanmadan yapıldı.Paddinglede aynısı yapabilridim...
    new_mails=[]
    for mail in mails:
        new_mail=[]
        for word in mail.split():
            if(len(new_mail) < 50 and word in json_tokenizer):
                new_mail.append(json_tokenizer[word])
        if(len(new_mail) < 50):
            zeros=list(np.zeros(50-len(new_mail) , dtype=int))
            new_mail=zeros + new_mail
        new_mails.append(new_mail)
    return np.array(new_mails , dtype=np.dtype(np.int32))  


train_set = make_token(x_train)

x_train[87]
y_train[87] #Normal mail

find_idx_json = json_tokenizer["merhaba"]

train_set[87]
test_set = make_token(x_test)
x_test[87]
test_set[87]

#%%MODEL OLUŞTURMA

model=Sequential()
embedding_size=50 
model.add(Embedding(input_dim=421, output_dim=embedding_size,input_length=50,name="embedding_layer"))

model.add(GRU(units=128,return_sequences=True)) #dropout ile overfittingi engelleyellim
model.add(GRU(units=64, return_sequences=True)) #bunu muhakkak dene
model.add(GRU(units=32))
model.add(Dense(1,activation="sigmoid"))

# optimizer = Adam(lr=1e-3)
model.compile(optimizer="Adam",loss="binary_crossentropy",metrics=["mse","accuracy"])
model.summary()

len(test_set)
len(y_train)

train_set=np.array(train_set)   
y_train=np.array(y_train)               
test_set=np.array(test_set)
y_test=np.array(y_test)

#MODELİ EĞİTME
history = model.fit(train_set,y_train,epochs=50,batch_size=128) 

#LOSS GRAFİK DEĞERLERİ

pyplot.plot(history .history ['accuracy'])
pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy Score')
pyplot.title('Accuracy GRAPHIC')
pyplot.show()

pyplot.plot(history .history ['mse'])
pyplot.xlabel('Epoch')
pyplot.ylabel('Mean Square Error')
pyplot.title('MSE GRAPHIC')
pyplot.show()

model.evaluate(test_set,y_test)                                                          

#MODELİ KAYDETME
model.save("last_model.h5")

#%% DIŞARDAN VERİ GİREREK MODELE PREDİCT ETTİRELİM TEST AMAÇLI

text1="geliyormusun okula ben gelmeyecemde"
text2="efsane fiyatlı indirimleri kaçırmayın samsung telefonlarda yuzde 50 varan indirim fırsatlarını kaçırma"
text3="google a0 analytics google analytics ayarlarını inceleyin size uygunolup olmadığını doğrulayın google analytics hesabınızı ekim 2018 mart 2020 arasında oluşturduysanız bizden önemli güncellemeleri almıyor olabilirsiniz posta güncellemelerini etkinleştirmenizi öneririz böylece kullanıma sunulan yeni özellikler performans önerileri google analytics teklifleri hakkında bilgi sahibi olabilirsiniz ayarlarinizi onaylayin"

liste=[text1,text2,text3]

test_set=make_token(liste)
# deneme_kumesi=tokenlestir(liste)
test_set[0]
test_set[1]
test_set[2]

result=model.predict(test_set)
# a=sum(result)
# a=float(a)
# a/len(sentencesvideo)

#%% MODELİ DÖNÜŞTÜRME YÜKLEME VE TEST ETME

import numpy as np
import pandas as pd
import json
from tensorflow.python.keras.models import load_model

new_model=load_model("last_model.h5")

with open('last_token.json') as json_dosyasi: 
    json_tokenizer=json.load(json_dosyasi)


def make_token(mails):   #Padding fonksyionu kullanmadan yapıldı.Paddinglede aynısı yapabilridim...
    new_mails=[]
    for mail in mails:
        new_mail=[]
        for word in mail.split():
            if(len(new_mail) < 50 and word in json_tokenizer):
                new_mail.append(json_tokenizer[word])
        if(len(new_mail) < 50):
            zeros=list(np.zeros(50-len(new_mail) , dtype=int))
            new_mail=zeros + new_mail
        new_mails.append(new_mail)
    return np.array(new_mails , dtype=np.dtype(np.int32))  

#%% DIŞARDAN VERİ GİREREK KAYDETTİĞİMİZ MODELE PREDİCT ETTİRELİM

text1="kişisel bakım süpermarkette i̇lk alişveri̇şi̇nle fiyat i̇ndi̇ri̇m kazan"
text2="efsane fiyatlı indirimleri kaçırmayın samsung telefonlarda yuzde 50 varan indirim fırsatlarını kaçırma"
text3="google a0 analytics google analytics ayarlarını inceleyin size uygunolup olmadığını doğrulayın google analytics hesabınızı ekim 2018 mart 2020 arasında oluşturduysanız bizden önemli güncellemeleri almıyor olabilirsiniz posta güncellemelerini etkinleştirmenizi öneririz böylece kullanıma sunulan yeni özellikler performans önerileri google analytics teklifleri hakkında bilgi sahibi olabilirsiniz ayarlarinizi onaylayin"
text4="fghfghfgh"

liste=[text1,text2,text3,text4]

pilot_train_set=make_token(liste)
new_result=new_model.predict(pilot_train_set)*100
print(new_result)

# yeni_sonuc=yeni_sonuc.tolist()
# yeni_sonuc=str(yeni_sonuc)

# for i in range(len(yeni_sonuc)):
#     for k in range(len(yeni_sonuc[i])):
        
#         if yeni_sonuc[i][k] == 0.5143779516220093:-
        
#             yeni_sonuc[i][k] = 0.0102323212312312
        

#%% Resimlerden Tahmin Sonucu Üretme

import pytesseract
import regex

text = pytesseract.image_to_string("a.png",lang="tur")  
liste=[text]

image_sentences=[]
image_regex=regex.metni_isle(liste[0])
image_sentences.append(image_regex)

# "split_stem" Fonksiyonunun Çalışması

image_split = split_stem(image_sentences)
    
image_train_set=make_token(image_split)
new_image_result=new_model.predict(image_train_set) * 100
new_image_result = float(new_image_result)
new_image_result = round(new_image_result, 4)


print("%" +str( new_image_result))
if new_image_result >= 50:
    print("Spam Mail") 
else:
    print("Not Spam Mail")






    
    

    
    






    
    
    
    
    
    
    
    
    
    
    