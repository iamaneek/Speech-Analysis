# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 05:19:33 2019

@author: ASUS
"""

import nltk

import os
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sp
from tqdm import tqdm
from textblob import TextBlob
from googletrans import Translator

from pandasql import sqldf


import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.

import matplotlib.pyplot as plt
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix





#This section converts audio file into text

with open("api-key.json") as sound_file:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS=sound_file.read()
    
r=sp.Recognizer()
sorted_files=sorted(os.listdir('speeches/'))

all_text=[]

for sound_file in sorted_files:
    name="speeches/"+sound_file
    with sp.AudioFile(name) as source:
        audio=r.record(source)
    text=r.recognize_google_cloud(audio,credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
    all_text.append(text)
    
''' THIS PART REPRESENTS TRANSCRIPTION OF THE SPEECHES
transcript = ""
for i, t in enumerate(all_text):
    seconds_total = i * 30
    m, s = divmod(seconds_total, 60)
    h, m = divmod(m, 60)

    # Format time as h:m:s - 30 seconds of text
    transcript = transcript + "{:0>2d}:{:0>2d}:{:0>2d} {}\n".format(h, m, s, t)

print(transcript)

with open("transcript_final.txt", "w") as f:
    f.write(transcript)
    

    
#reading from transcripts and breaking down into each sentences
with open("transcript_final.txt", "r") as in_file:
    in_text = in_file.read()
    sents = nltk.sent_tokenize(in_text)

print(sents)
'''

#SENTIMENT ANALYSIS

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    

sentiment_analyzer_scores("Hi, how are you?")
sentiment_analyzer_scores("It feels great to talk to you after such a long time.")
sentiment_analyzer_scores("Oh my god, look at you.")
sentiment_analyzer_scores("You know, I am finally feeling happy.")
sentiment_analyzer_scores("Hi, good to see you.")



score_list=[] 
positive_score=[]
negative_score=[]
neutral_score=[]
compound_score=[]

for i in all_text:
    score=analyser.polarity_scores(i)
    print(score)
    positive_score.append(score['pos'])
    negative_score.append(score['neg'])
    neutral_score.append(score['neu'])
    compound_score.append(score['compound'])

score_tuples=list(zip(all_text,positive_score,neutral_score,negative_score,compound_score))

score_list=pd.DataFrame(score_tuples,columns=['Text','Positive','Neutral','Negative','Compound'])


#VOICE ANALYSIS

sound_intensity=[]
loaded_audios=[]
state_voice=[]
sentiment_state=[]


pathAudio = "actors/"
files = librosa.util.find_files(pathAudio) 
files = np.asarray(files)
for y in files: 
    data,sr = librosa.load(y, sr = 16000,mono = True)  
    loaded_audios.append(librosa.stft(data))


for i in loaded_audios:
    Xdb=librosa.amplitude_to_db(abs(i))
    mean_intensity=abs(np.mean(abs(Xdb)))
    sound_intensity.append(mean_intensity)

#sound intensities

sound_intensity

intensity=pd.DataFrame(sound_intensity,columns=['Intensity'])


for k in intensity['Intensity']:
    if k<20:
        state="whisper"
        state_voice.append(state)
    elif (k>=20 and k<=40):
        state="low"
        state_voice.append(state)
    elif (k>=40 and k<=60):
        state="normal"
        state_voice.append(state)
    elif k>60:
        state="excited"
        state_voice.append(state)



for m in score_list['Compound']:
    if(m>=-1 and m<=-0.5):
        senti_state="very negative"
        sentiment_state.append(senti_state)
    elif (m>-0.5 and m<-0.05):
        senti_state="negative"
        sentiment_state.append(senti_state)
    elif (m>=-0.05 and m<=0.05):
        senti_state="neutral"
        sentiment_state.append(senti_state)
    elif (m>0.05 and m<=0.5):
        senti_state="positive"
        sentiment_state.append(senti_state)
    elif (m>0.5 and m<=1):
        senti_state="very positive"
        sentiment_state.append(senti_state)

state_table=pd.DataFrame(state_voice,columns=['State'])
sentistate_table=pd.DataFrame(sentiment_state,columns=['Sentiment'])
#overall DataFrame

analysis_table=pd.concat([score_list,intensity,state_table,sentistate_table],axis=1)


'''
df = DataFrame(analysis_table,columns=['Compound','Intensity'])
df.plot(x ='Intensity', y='Compound', kind = 'line')
'''

'''
analysis_table['Positive'].plot(label='pos',figsize=(10,5))
analysis_table['Neutral'].plot(label='neu')
analysis_table['Negative'].plot(label='neg')
analysis_table['Compound'].plot(label='compound')
plt.xlabel('Number of Speeches',fontsize=14)
plt.title('Sentiment Presence in Speeches',fontsize=16)
plt.legend()



compound_percent=analysis_table['Compound']*100

analysis_table['Intensity'].plot(label='Intensity')
compound_percent.plot(label='compound')
plt.xlabel('Number of Speeches',fontsize=14)
plt.title('Intensity v/s Sentiment in Speeches',fontsize=16)
plt.legend()

positive_percent=analysis_table['Positive']*100

'''


pysqldf = lambda q: sqldf(q, globals())

nonzero_positive=pysqldf("Select * from analysis_table where Positive!= 0")
nonzero_compound=pysqldf("Select * from analysis_table where Compound!= 0")


(nonzero_positive['Intensity']*100/110).plot(label='Intensity')
(nonzero_positive['Positive']*100).plot(label='Positive')
plt.xlabel('Number of Speeches',fontsize=14)
plt.title('Intensity V/S Positive ',fontsize=15)
plt.legend()


(analysis_table['Intensity']).plot(label='Intensity')
(analysis_table['Compound']*100).plot(label='Sentiment')
plt.xlabel('Number of Speeches',fontsize=14)
plt.title('Intensity V/S Overall Sentiment',fontsize=15)
plt.legend()


nonzero_compound['Positive'].plot(label='pos',figsize=(10,5))
nonzero_compound['Neutral'].plot(label='neu')
nonzero_compound['Negative'].plot(label='neg')
#analysis_table['Compound'].plot(label='compound')
plt.xlabel('Number of Speeches',fontsize=14)
plt.title('Sentiment Presence in Speeches',fontsize=16)
plt.legend()


fig=plt.figure(figsize=(10,6))
ax1=fig.add_axes([0,0,0.75,0.75])
#ax1.set_title('Intensities in Audio Samples',size=16)
ax1.plot(analysis_table['Intensity'],color='green')
plt.xlabel('Number of Samples',size=14)
plt.ylabel('Sound Intensities',size=14)
plt.show()    


fig1=plt.figure(figsize=(15,7))

ax2=fig1.add_subplot(231)
ax2.plot(analysis_table['Positive'],color='green')
plt.ylabel('Positive Score',size=14)


ax2=fig1.add_subplot(232)
ax2.plot(analysis_table['Neutral'],color='blue')
plt.ylabel('Neutral Score',size=14)


ax2=fig1.add_subplot(234)
ax2.plot(analysis_table['Negative'],color='red')
plt.xlabel('Audio Samples',size=14)
plt.ylabel('Neutral Score',size=14)


ax2=fig1.add_subplot(235)
ax2.plot(analysis_table['Compound'],color='teal')
plt.xlabel('Audio Samples',size=14)
plt.ylabel('Overall Score',size=14)

