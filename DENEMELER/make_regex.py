from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


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
    metin = metin.lower()

    
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