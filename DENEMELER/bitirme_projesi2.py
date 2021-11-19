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
from tensorflow.python.keras.layers import Dense,GRU,Embedding,CuDNNGRU,Dropout
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


#%%TRAİN TEST SPLİT

abc=split_stem(sentences)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(abc,target,test_size=0.15,random_state=0)
    
#%% TOKENLEŞTİRME

num_words=550
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(abc)
tokenizer.word_index

tokenizer.word_index.items() # 3129 tane kelimemiz var totalde

x_train_tokens=tokenizer.texts_to_sequences(x_train)
x_test_tokens=tokenizer.texts_to_sequences(x_test)


num_tokens=[len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens=np.array(num_tokens)

np.mean(num_tokens)  # yorumlarda toplam 10.24 tane token bulunuyor.
np.max(num_tokens)  # en fazla 53 tokens bulunuyor.
np.argmax(num_tokens)
x_train[136]  # en uzun token bu olmus oluyor...        

max_tokens=np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens=int(max_tokens) #networkümüze 25 tokenli vektörler verelim...

np.sum(num_tokens<max_tokens)/len(num_tokens) # yorumların %94 ünde 25 ten az tokenler bulunuyor yani 0ekleyerek 25 e çıkaracaz % 6 sında fazladan bulunan tokenleri atıcaz
x_train_pad= pad_sequences(x_train_tokens,maxlen=max_tokens) #artık her yorum 25 vektör uzunlugunda olacak
x_test_pad= pad_sequences(x_test_tokens,maxlen=max_tokens)

x_train_pad.shape # 335 tane spam mail var
x_test_pad.shape

np.array(x_train_tokens[100])
x_train_pad[100]


#%% burda maillerdeki en cok gecen kelimeleri görebiliyoruz...
idx=tokenizer.word_index
inverse_map=dict(zip(idx.values(),idx.keys()))
def tokens_to_string(tokens):
    words=[inverse_map[token] for token in tokens if token!=0]
    text=' '.join(words)
    return text


x_train[101]
tokens_to_string(x_train_tokens[101])


#%%Word2Vec modeli...
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
  

#%%
#TRAİN TEST SPLİT

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(sentences,target,test_size=0.20,random_state=0)

# len(x_train)
# len(x_test)

# #Orjinal verisetimizdeki mail
# dataMail[4]
    
# #Veri önişlemeye girmiş mail
# # metni_isle("Tüm Tabletlerde KDV tutarı kadar indirim ve SAMSUNG 49NU7100 124 Ekran 4K UHD TV 3597TL yerine 2998TL.Son gün: 2 Eylül.Stoklarla sınırlı fırsatları kaçırmayın.Detaylar:Teknosa Mağazaları'nda ve https://bit.ly/2HwYv9k .SMS ret için TSA RET yazıp 7889'a ücretsiz gönderebilirsiniz.")

# x_train[4]
# y_train[4]

# import json
# with open('last_token.json') as json_dosyasi: #Daha önceden oluşturdugum "tokenizer.json" dosyasını "json_tokenizer" olarak yüklüyorum.
#     json_tokenizer=json.load(json_dosyasi)
    
# # json_tokenizer[""]

# # "tokenlestir" fonksiyonu
# def make_token(mails):   #Padding fonksyionu kullanmadan yapıldı.Paddinglede aynısı yapabilridim...
#     new_mails=[]
#     for mail in mails:
#         new_mail=[]
#         for word in mail.split():
#             if(len(new_mail) < 50 and word in json_tokenizer):
#                 new_mail.append(json_tokenizer[word])
#         if(len(new_mail) < 50):
#             zeros=list(np.zeros(50-len(new_mail) , dtype=int))
#             new_mail=zeros + new_mail
#         new_mails.append(new_mail)
#     return np.array(new_mails , dtype=np.dtype(np.int32))  


# train_set = make_token(x_train)

# x_train[87]
# y_train[87] #Normal mail

# find_idx_json = json_tokenizer["merhaba"]

# train_set[87]
# test_set = make_token(x_test)
# x_test[87]
# test_set[87]

#%%MODEL OLUŞTURMA

model=Sequential()
embedding_size=50 


model.add(Embedding(input_dim=num_words, output_dim=embedding_size,input_length=max_tokens,name="embedding_layer"))

model.add(GRU(units=128,return_sequences=True)) #dropout ile overfittingi engelleyellim
model.add(GRU(units=64, return_sequences=True)) #bunu muhakkak dene

model.add(GRU(units=32))
model.add(Dropout(0.8))
model.add(Dense(1,activation="sigmoid"))

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["mse","accuracy"])
model.summary()

x_train=np.array(x_train)   
y_train=np.array(y_train)               
x_test=np.array(x_test)
y_test=np.array(y_test)

#MODELİ EĞİTME
history = model.fit(x_train_pad,y_train,epochs=50,batch_size=128) 

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

result=model.evaluate(x_test_pad,y_test)     
print(result[1])                                                     

#MODELİ KAYDETME
model.save("last_model4.h5")


#%%
y_pred=model.predict(x=x_test_pad[0:100])
y_pred=y_pred.T[0]

cls_pred=np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
cls_true=np.array(y_test[0:100])

incorrect=np.where(cls_pred != cls_true)
incorrect=incorrect[0]
len(incorrect) # 100 tane yorum içerisinden 14 tanesi yanlıs bilinmiş???
idx=incorrect[0] #8 verdi

text=x_test[idx]  # bu mail spammi spam değilmi bilemedi mesela
y_pred[idx]



#%% DIŞARDAN VERİ GİREREK MODELE PREDİCT ETTİRELİM TEST AMAÇLI

text1="geliyormusun okula ben gelmeyecemde"
text2="efsane fiyatlı indirimleri kaçırmayın samsung telefonlarda yuzde 50 varan indirim fırsatlarını kaçırma"
text3="google a0 analytics google analytics ayarlarını inceleyin size uygunolup olmadığını doğrulayın google analytics hesabınızı ekim 2018 mart 2020 arasında oluşturduysanız bizden önemli güncellemeleri almıyor olabilirsiniz posta güncellemelerini etkinleştirmenizi öneririz böylece kullanıma sunulan yeni özellikler performans önerileri google analytics teklifleri hakkında bilgi sahibi olabilirsiniz ayarlarinizi onaylayin"
text4="sen mükemmel bir insansın demi"
text5="bu efsane indirim için daha neyi bekliyorsunki"
text6="nasılsın la nasıl gidiyor ben sıkılyırum"
text7="merhaba nasılsınız acaba nasıl gidiyor"
text8="dışarı çıkmayı düşünmüyorum"
text9="aşırı sıkılıyorum ne yapalımki bir"
liste=[text1,text2,text3,text4,text5,text6,text7,text8,text9]

tokens=tokenizer.texts_to_sequences(liste)

tokens_pad=pad_sequences(tokens,maxlen=max_tokens)
tokens_pad.shape


result=model.predict(tokens_pad)*100
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
import make_regex

text = pytesseract.image_to_string("a.png",lang="tur")  
liste=[text]

image_sentences=[]
image_regex=make_regex.metni_isle(liste[0])
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






    
    

    
    






    
    
    
    
    
    
    
    
    
    
    