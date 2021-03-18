# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:32:59 2021

@author: Mina.Melek
"""

from utilities import * 
# import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.optimizers import Adam, Adamax
import re
sns.set_theme(style="darkgrid")

class Evaluate(object):
    def __init__(self, data, text_column='message'):
        """Construct the Evaluation class.

        Keyword arguments:
            data -- a Pandas DataFrame containing the data samples to be processed and evaluated (must be of the type -> 'pandas.core.frame.DataFrame')
            text_column -- a string represents the name of the target column that contains the text data  (default 'message')
        """

        if not isinstance(data, pd.core.frame.DataFrame): raise TypeError("data must be a \"pandas dataframe\"")
        if not text_column in data.columns: raise ValueError("You didn't specify the correct column for the text data in the input dataframe")

        self.__version__ = '2.0.2'
        self.data = data.copy()
        self.message = self.data[text_column].astype(str).copy()
        self.vect_type = None
        self.word_vectorizer = None
        self.model_parameters = None
        self.predictions = None
        self.pred_count = 0
        self.majority_index = -1
        
        self.preprocess()
        
        
        

    def preprocess(self):
        """Prepreocess the text message before evaluation"""
        
        # function loops over a sentence removing repeated consequent phrases
        def rep(x):
            while True:
                d = len(x)
                x = re.sub(r'\b([\w\s]+)( \1\b)+', r'\1', x)
                if len(x) == d:
                    return x
                
        # cleaning message
        self.clean(pipelined=True);
        # remove repeated sentences in text
        self.message = self.message.apply(rep)
        # #remove samples with only numbers in the text
        # self.message = self.message[self.message.map(lambda x: not bool(re.match(r'^[\d\s]+$', x)))]
        # # Remove empty strings after cleaning
        # self.message = self.message[self.message.map(len)>=3]
        # # remove duplicates
        # self.message = self.message.drop_duplicates().dropna()
        
        self.data = self.data.loc[self.data.index.isin(self.message.index)]
        self.data['cleaned'] = self.message
        self.data.reset_index(inplace=True, drop=True)
        self.message = self.data['cleaned']
        
    def get_lists(self):
        """return topic modeling lists"""
        
        Medical_conslut = 'عندي علاج الم الاسنان العلاج القولون الركبه اعمل عندي الم القلب شديد الحل عندي عمليه وعندي اسنان المعده خشونه الشمال عندي القولون وعملت الم شديد الركبة عملت عملية مفصل العصبي والم الكتف الظهر اسفل المعدة الالم علاج القولون العصبي اليمين التهاب ارتجاع خشونة عندي خشونه طبيعي الاعراض الصدر البطن عنده مشكله سكر تحت غضروف وفي كتفي مشكلة الايسر العمليه مين قولون الركبتين قلب تنميل رجلي عندي ارتجاع ضعف الام بطني نتيجه بحس بكام اسفل الظهر ضغط ورم الرقبه تركيب السكر منظار علاج طبيعي مزمن اية عندي التهاب مستمر بدون باستمرار الغضروف صدري خشونه الركبه ضرس تحليل عندي عندي مشكله النوم العملية حرقان بقالي زراعة التهابات اليمني حقن جامد امساك ايدي اريد قطع وباخد امشي عندها بنتي الكلي عضلة واخدت عندي قولون الحاله عظام والقولون القدم ابني اعاني اليسري مجهود اشعه اسناني رنين طب شديدة تعبانه طبيب دم عندي خشونة فايده علشان الضغط الدم شويه عضله والام كبير امراض واخذت باخد تغير عصبي اقل البول عالي العقل فترة مرض تقويم وارتجاع بسيط خالص مريض فتره الفقري فوق تعبني بعاني وغازات قلبي وزني الكبد عضلة القلب المشي الحوض الايمن فقرات انتفاخ المفاصل ضربات جراحه الوزن ابغي مني علاج عندي امساك الغدة وعايز اثناء عندي مشكلة عندي حرقان عندي عندي اسنان تعب للقولون دلوقتي العظام جسمي دواء الجسم اليد وانتفاخ الفقرات ضربات القلب جراحة الرقبة الثه واخدت علاج الطبيب والعلاج تمزق عضله القلب وسكر عندي اللثة بسبب جامده وضغط فايده التنفس خفقان العصب تلبيس عندي خلع سبب مستشفي عندي قطع المرئ العمود اعراض عندي عندي التهابات ركبتي تاكل الساقين كسر شديده مشاكل عندي الام وكمان عندي سكر اخدت عندي الم الركبه شرخ تضخم الاعصاب اخذ عمليات دهون القدمين علاج جلسات عندي ورم عندي غضروف وامساك وبحس عندي عدم واعصاب مخ التحليل مخ واعصاب اسهال تعبان الرحم العمود الفقري كشفت والكتف المفصل سكر وضغط اليمن وتعبت العلوي لسه السلم حشو الورم وطلع شراين الركبه الشمال مفاصل الحمام انسداد زرع ارتفاع واخذت علاج البراز فظيع لحد نزيف تعبت تغير مفصل تعمل والتهابات الاسنان زراعه الالام بيجي عصب تنظيف الدوالي دائم بالم واخد اقل مجهود الفك رقبتي جرثومة عندي ضغط عندي اعلي اطباء الغده وجود دوالي الاثنين الحمل الفخد هضم عندي تنميل الشريان حمل الشعر علاج ضعف اشاعه حامل تحاليل النسا صوت عارفة اعمل عمليه يدي الدرقية عندي مفصل الركبه علاج القولون الصمام القولون وعندي علاجه كثير اورام خارج مرات الشراين وتورم وضعف علاج مثل علاج الخلف شديد مفتوح السكري الطبيعي ماهو علاج الوجع كيس ادويه مسالك تعباني وكشفت الرباط وضيق مفصل الحوض التنفس عندي فم الكعب معدتي والالم المراره ضعف عضلة تكميم حالتي فهل علاج الفقره كيلو اناعندي'
        service_support = 'تامين عندكم عندكم تامين التعاونيه تامين التعاونيه لديكم تامين التامين تامين تامين عندكم يقبل التعاونية يشمل تامين التعاونية عندكم يغطي تقبلو شركة عندكم تقبلو تامين عندكم الراجحي تامين تامين صحي شركات تقبلون تامين صحي تقبلون تامين متوفر ميد تامين ميد تقبل كلاس تتعاملوا عندكم عندكم تبع تكافل يقبل تامين فئة تامين كلاس تامين تكافل مقبول تكافل الراجحي تعاقد تابع عندكم تتعاملون تقبلوا عندكم متاح عندكم لديكم جهاز بطاقة شركه عندكوا خصم للتامين ميد تامين بالتقسيط عندكم ميد متعاقدين التامين عندكم شركات التامين عيادة عندكن اسوي توجد الصحي موجوده التامين الصحي جهاز بتامين اماكن الكورونا استفيد التعامل ارغب بالتامين كورونا لشركة التامينات عيادات اختصاص الخاصة الصحه تبع التامين استفسر تسجيل كرونا ضمن انتو الهضمي بتتعاملوا تتعاملو التامين العروض التامين الشركات استعلم هل للشركات التسجيل'
        PriceBooking_inquiry ='رقم مواعيد وكم علاج احجز السعر حجز مواعيد التواصل وبكم وكم السعر كام عروض مواعيد والاسعار موعد الحجز العرض ارقام الخدمة احجز بكام عرض تليفون سعرها تكلفة وكام تتكلف التليفون رقم تليفون رقم التليفون للحجز لحجز تكلفتها سعر خدمة التكلفة التواصل رقم حجزت تلفون عاوز التكاليف العرض رقم رقم تلفون تكلف معرفة اسعار الموعد سعر الهاتف وكم تكلفتها احجز جوال متواجد احجز موعد سجلت العرض مناسب رقم جوال رقم الهاتف يكلف جوالي رقم جوالي اتواصل ميعاد ماهي عاوز وقت سعره خصومات عايزه تكلفة بدي الرقم اخر زرع التركيبات تكلفه ارسال رقم سعرها كام التكلفه مبلغ كام كم بكم تفاصيل التفاصيل تكلف كلف سعر تليفون رقم مواعيد نحجز'
        place_inquiry = 'فين المكان مكان المكان فين عنوان فين المكان المركز جده فين مكان وين العنوان مكانها عنوان فين مصر القاهرة مكانكم الموقع موقعكم جدة موقع فضلك فين موقع فين مكانها اين مكان فين الموقع مكان عنوانكم العنوان فين مكان الاسكندرية القاهره وين المكان مدينة اسكندريه والمكان اندلسيه الاندلسية السعودية حي سموحة بجده الجامعه مكه اسكندرية حي الجامعه فرعكم فروعكم الفرع الشلالات واين حي سموحة المعادي بحي الرياض المكرونه حي مصر الاندلسيه بالقاهرة وفين سموحه الاسكندريه وين الموقع وين موقعكم المعادي عنوان مكان فرع فين موقع وين التحلية'

        obj_lists = {'Medical_consult': Medical_conslut,
                     'support': service_support,
                     'PriceBooking': PriceBooking_inquiry,
                     'place': place_inquiry}
        
        thresholds = dict(zip(obj_lists.keys(), [0.0033, 0.01, 0.01, 0.015]))
        
        return obj_lists, thresholds


    def get_data(self):
        """returns the data DF"""

        return self.data



    def clean(self, input=None, add_to_df=False, pipelined=False):
        """apply text cleaning steps to the message column
        
        Keyword arguments:
            input -- the input series to be processed (default None)
                **Note** - if the value isn't None, it's expected to be used in a pipeline. 
            add_to_df -- a boolean value; if true, it adds the result as a seprate column to the dataframe (default False)
            piplined -- a boolean value; if true, it overwrite self.message with the new transformation for a pipeline (default False) 
        """

        if input is None:
            input = self.message
        cleaned = clean_df(input) # Function defined in utilities.py
        if add_to_df:
            self.data["cleaned"] = cleaned
        if pipelined:
            self.message = cleaned
        return cleaned

    def stem(self, input=None, add_to_df=False, pipelined=False):
        """apply stemming to the message column
        
        Keyword arguments:
            input -- the input series to be processed (default None)
                **Note** - if the value isn't None, it's expected to be used in a pipeline. 
            add_to_df -- a boolean value; if true, it adds the result as a seprate column to the dataframe (default False)
            piplined -- a boolean value; if true, it overwrite self.message with the new transformation for a pipeline (default False) 
        """

        if input is None:
            input = self.message
        stemmed = stem(input) # Function defined in utilities.py, clean text before stemmed
        if add_to_df:
            self.data["stemmed"] = stemmed
        if pipelined:
            self.message = stemmed
        return stemmed

    def lemmatize(self, input=None, add_to_df=False, pipelined=False):
        """apply lemmatization to the message column
        
        Keyword arguments:
            input -- the input series to be processed (default None)
                **Note** - if the value isn't None, it's expected to be used in a pipeline. 
            add_to_df -- a boolean value; if true, it adds the result as a seprate column to the dataframe (default False)
            piplined -- a boolean value; if true, it overwrite self.message with the new transformation for a pipeline (default False) 
        """
        # takes too much time #FIXED
        if input is None:
            input = self.message
        lemmatized = lemma(input) # Function defined in utilities.py, clean text before lemmatization
        if add_to_df:
            self.data["lemmatized"] = lemmatized
        if pipelined:
            self.message = lemmatized
        return lemmatized

    def remove_stopwords(self, input=None, add_to_df=False, pipelined=False, stopWords=None, split=False, progress_per=None):
        """remove stopwords from the message column
        
        Keyword arguments:
            input -- the input series to be processed (default None)
                **Note** - if the value isn't None, it's expected to be used in a pipeline. 
            add_to_df -- a boolean value; if true, it adds the result as a seprate column to the dataframe (default False)
            piplined -- a boolean value; if true, it overwrite self.message with the new transformation for a pipeline (default False) 
        """

        if input is None:
            input = self.message
        no_sw = remove_stopwords(input, stopWords, split, progress_per) # Function defined in utilities.py
        if add_to_df:
            self.data["without_stopwords"] = no_sw
        if pipelined:
            self.message = no_sw
        return no_sw
    
    def eval_lexicons(self, lex_path, threshold=0):
        """evaluate a lexicon-based prediction on the data, the lexicons are a dictionary of words labeled according to their polarity; 
        ideally: positive sentiment words have a polarity of +1 and negative sentiment words have a polarity of -1
        
        Keyword arguments:
            lex_path -- a string represents a path to the file.csv containing lexicons (polarity should be in range [-1 : 1])
            threshold (optional) -- an int value represents a threshold between the positive and negative labels that define the difference in polarity (default 0)
        """
        
        assert lex_path.endswith('.csv'), "You must enter a valid path to a lexicons csv file"
        lex = pd.read_csv(lex_path, index_col=0)
        return self.message.map(lambda x: sum([lex.loc[w].polarity if w in lex.index else threshold for w in get_all_ngrams(x)])).map(lambda x: 'pos' if x>threshold else 'neg' if x<threshold else 'obj')
    
    def pred_majority(self, majority_index=-1):
        """calculate the majority voting for multiple predictions, incase more than one model was used for prediction
        Keyword arguments:
            majority_index -- an integer value represent the index of predictions column that should be selected in case of no majority (default -1)
        """
        
        assert majority_index < self.pred_count+1, "the value of 'majority_index' is {}, which is higher than number of predictions".format(majority_index)
        def maj(x):
            if max(map(x.count, x)) == 1:
                return x[majority_index]
            return max(x, key=x.count)

        return self.predictions.apply(lambda x: maj(x.tolist()), axis=1)
        
    def transform(self, vect_type="w2v", vect_path=None, vect_model=None):
        """vector transform message column for classification

        Keyword arguments:
            vect_type -- a string value describes the vectorization method used for preparing the data. 
                it can only take either 'tfidf' -> tfidf vectorization, or 'w2v' -> word2vec vectorization (default 'w2v')
            vect_path -- The path of the vectorizing model, it must match the name specified in 'vect_type' (default None)
            vect_model (optional) -- an object of the vectorizing model can be given here to save loading time, it must be the same type specified in 'vectorized' (default None)
        """

        self.vect_type = vect_type
        if vect_type=='tfidf':
            if vect_path is None:
                vect_path = './models/tfidf_features.pkl'

            # Loading vectorizing model
            print("Loading Tf-Idf model ...")
            if vect_model is None:
                with open(vect_path, 'rb') as fid:
                    vocabulary = pickle.load(fid)
                    idfs = pickle.load(fid)
                tfidf = TfidfVectorizer(tokenizer=str.split, lowercase= False, ngram_range=(1,2), max_features=5000, vocabulary=vocabulary )#, max_df=0.7, min_df=5)#, preprocessor=sents, stop_words=cachedStopWords)
                tfidf.idf_=idfs
            else:
                tfidf = vect_model

            # Transformation
            print("Transforming data ...", end=' ')
            X = self.stem()
            X = self.remove_stopwords(X)
            X = tfidf.transform(X).toarray()

        elif vect_type=='w2v':
            if vect_path is None:
                vect_path = './models/w2v_model.bin'

            # Loading vectorizing model
            print("Loading Word2Vec model ...")
            if self.word_vectorizer:
                pass
            elif vect_model is None:
                w2v_model = KeyedVectors.load_word2vec_format(vect_path, binary=True) if vect_path.endswith('.bin') else Word2Vec.load(vect_path).wv
                self.word_vectorizer = WordVecVectorizer(w2v_model) # Class defined in utilities.py
            else:
                self.word_vectorizer = vect_model

            # Transformation
            print("Transforming data ...", end=' ')
            X = self.clean()
            X = self.word_vectorizer.transform(X)

        else:
            raise ValueError("vect_type should only be either 'tfidf' or 'w2v', however you entered {}".format(vect_type))
        print("Done.")

        return X

    def predict(self, input=None, model_name='FCNN_w2v_model', lexicon_prediction=False, lex_path=None):
        """Evaluate the FCNN model and produce predictions

        Keyword arguments:
            input -- a numpy array represent the transformed data; it should be the output of self.transform in the shape (# of samples, features_dim) (default None). 
            model_name -- a string represents the name of the folder, which contains the model, the full path would be './models/model_name' (default 'FCNN_w2v_model)
            lexicon_prediction -- a boolean value indicates the use of lexicons for prediction instead of model evaluation (default False)
            lex_path --  a string represents a path to the file.csv containing lexicons; used when lexicon_prediction = True (default None)
        """

        if lexicon_prediction:
            print("Loading Lexicons ... ")
            lex_path = './models/Full_lexicons.csv' if lex_path is None else lex_path
            predictions = self.eval_lexicons(lex_path)
            self.vect_type = 'lexicons'
        else:
            # in case the input is not given
            if input is None:
                input = self.transform("w2v")
            # Loading the classification model
            print("Loading Classification model ... ")
            model, parameters = load_keras_model(model_name) # Function defined in utilities.py
            # Defining parameters
            self.model_parameters = parameters
            dense_layers, opt_name, batch_size, lr, decay, int_category = self.model_parameters['param']
            cats = len(int_category)
            input_shape = (input.shape[1], )
            if opt_name=='Adamax':
                opt = Adamax(lr=lr, decay=decay)
            else:
                opt = Adam(lr=lr, decay=decay)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            # Evaluating the model
            print("Evaluating the model ...", end=' ')
            predictions = model.predict(input).argmax(axis=1)
            predictions = list(map(lambda x: int_category[x], predictions))
        
        predictions = pd.Series(predictions, name=self.vect_type)
        if self.predictions is None:
            self.predictions = predictions
            self.data['predicted'] = self.predictions.values
        else:
            self.predictions = pd.concat([self.predictions, predictions], axis = 1)
            if self.vect_type=='w2v': self.majority_index = self.pred_count
            self.data['predicted'] = self.pred_majority(self.majority_index)
        
        self.pred_count += 1
        print("Done.")

        return self.data['predicted']

    def add_dicts(self):
        """Categorize topics for the text data with respect to a dictionary"""

        # a pre-requisit
        if "predicted" not in self.data.columns:
            self.predict(model_name='FCNN_'+self.vect_type+'_model')

        print("Adding dictionaries ... ", end=' ')
        
        def jaccard_similarity(query, document):
            intersection = set(query.split()).intersection(set(document.split()))
            union = set(query.split()).union(set(document.split()))
            return len(intersection)/len(union)
        
        self.data["tags"] = ""
        self.data["inquiry"] = 0
        self.data["ALL_Categories"] = ''

        stemmed = self.stem()
        lemmed = self.lemmatize() # takes too much time # FIXED
        topics, thresholds = self.get_lists()

        for idx in self.data.index:

            # Adding "tags" and "inquiry" columns
            sentence = stemmed.loc[idx].split() + lemmed.loc[idx].split()
            if any(e in sentence for e in ['ماد' ,'دفع' ,'مبلغ','مبالغ', 'ثمن','تمن', 'كام' ,'كم', 'بكم', 'كلف', 'سعر', 'جنيه', 'ريال', 'خصم', 'غال', 'فلس', 'مجان','مجا', 'رخص']): self.data.at[idx, 'tags'] += 'price-'; self.data.at[idx, 'inquiry'] = 1
            if any(e in sentence for e in ['خدم', 'دري', 'مدير', 'ادار', 'نتظار', 'نتظر',  'شخيص' ,'طويل', 'استقبال' ,'معامل']): self.data.at[idx, 'tags'] += 'service-'; self.data.at[idx, 'inquiry'] = 1
            if any(e in sentence for e in ['موقف', 'مواقف', 'مصعد', 'دور', 'مبن', 'زدحام', 'زحم']): self.data.at[idx, 'tags'] += 'structure-'; self.data.at[idx, 'inquiry'] = 1
            if 'تامين' in self.message.loc[idx]: self.data.at[idx, 'tags'] += 'insurance-'; self.data.at[idx, 'inquiry'] = 1
            if any(e in sentence for e in ['ساع', 'مت', 'معاد', 'ميعاد', 'موعد', 'مواعيد', 'وقت' ,'تخر']): self.data.at[idx, 'tags'] += 'datetime-'; self.data.at[idx, 'inquiry'] = 1
            if any(e in sentence for e in ['موجود', 'عنو', 'روح', 'مك', 'مكان', 'فرع', 'فين', 'موقع', 'وين']): self.data.at[idx, 'tags'] += 'place-'; self.data.at[idx, 'inquiry'] = 1
            for k, v in topics.items():
                if jaccard_similarity(self.message.loc[idx], v) >= thresholds[k]:
                    if 'price-' in self.data.loc[idx, 'tags'] and k=='PriceBooking':
                        continue
                    self.data.at[idx, 'tags'] += k+'-'; self.data.at[idx, 'inquiry'] = 1 
            if any(e in sentence for e in ['عند' ,'عا' ,'علاج' ,'كيف', 'علم', 'رجاء', 'محتاج', 'عايز', 'عاوز', 'تفاصيل', 'رجو', 'عرف', 'هل', 'ايش', 'ايه', 'ازي', 'معلش', 'سمح', 'مين', 'ياري', 'مكن', 'ليه']): self.data.at[idx, 'inquiry'] = 1 # general
            
            # Adding "ALL_Categories" column
            if self.data.predicted[idx] in ['pos', 'neg']: #if self.data.at[idx, 'inquiry']==0:
                self.data.at[idx, 'ALL_Categories'] = 'Positive-Feedback' if self.data.predicted[idx]=='pos' else 'Negative-Feedback'# if self.data.predicted[idx]=='neg' else 'Other'
            elif self.data.at[idx, 'inquiry']==1:
                if self.data.at[idx, 'tags']=='price-': self.data.at[idx, 'ALL_Categories'] = 'Price-Inquiry'
                elif self.data.at[idx, 'tags']=='place-': self.data.at[idx, 'ALL_Categories'] = 'Place-Inquiry'
                elif self.data.at[idx, 'tags']=='datetime-': self.data.at[idx, 'ALL_Categories'] = 'Appointments-Inquiry'
                elif self.data.at[idx, 'tags']=='insurance-': self.data.at[idx, 'ALL_Categories'] = 'Insurance-Inquiry'
                elif self.data.at[idx, 'tags']=='service-': self.data.at[idx, 'ALL_Categories'] = 'Services-Inquiry'
                elif self.data.at[idx, 'tags']=='Medical_consult-': self.data.at[idx, 'ALL_Categories'] = 'Medical-Consultation'
                elif self.data.at[idx, 'tags']=='support-': self.data.at[idx, 'ALL_Categories'] = 'Insurance-Inquiry'
                elif self.data.at[idx, 'tags']=='PriceBooking-': self.data.at[idx, 'ALL_Categories'] = 'Booking-Inquiry'
                elif self.data.at[idx, 'tags']=='' : self.data.at[idx, 'ALL_Categories'] = 'General-Inquiry' #np.isnan(self.data.at[idx, 'tags'])
                else: self.data.at[idx, 'ALL_Categories'] = 'Mixed-Inquiries'
            else:
                self.data.at[idx, 'ALL_Categories'] = 'General Conversation'
        
        print("Done.")

    def visualize(self, kind='pie'):
        """Visualizing the final data; making tow pie plots for feedback data, inquiry data
        Keyword arguments:
            kind -- specifies the type of plot that should be plotted, 
                takes as values ('pie' or 'bar') each represent a part of the prediction results (default 'pie') 
        """
        
        # a pre-requisit
        if "ALL_Categories" not in self.data.columns:
            self.add_dicts()
        
        print("Visualizing ...")
        if kind=='pie':
            fig1, ax_pie = plt.subplots(1, 2, figsize=(13, 26))
            
            FB = ["Positive-Feedback", "Negative-Feedback"]
            feedback = self.data.query('ALL_Categories in @FB').ALL_Categories.value_counts()
            ax_pie[0].set_title('Categories\' feedback Percentage', fontsize=12)
            feedback.plot(kind='pie', labels=feedback.index,
                        wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                        textprops={'fontsize': 10}, ax=ax_pie[0])
            
            inquiries = self.data.query('ALL_Categories not in @FB').ALL_Categories.value_counts()
            ax_pie[1].set_title('Categories\' inquiries Percentage', fontsize=12)
            inquiries.plot(kind='pie', labels=inquiries.index,
                                        wedgeprops=dict(width=.7), autopct='%1.0f%%', startangle= -20, 
                                        textprops={'fontsize': 10}, ax=ax_pie[1])

        if kind=='bar':
            fig2, ax_bar = plt.subplots(2, 1, figsize=(9, 12))
            
            NEG = self.data.query("predicted=='neg'")
            negative_reason = NEG.tags.str.rstrip('-').str.split('-').explode().replace({'': 'General Service Complaints',
                                                                                         'price': 'High Prices',
                                                                                         'service': 'Specified Service',
                                                                                         'datetime': 'Appointments',
                                                                                         'place': 'Location-Related',
                                                                                         'structure': 'InfraStructure-Related',
                                                                                         'insurance': 'Insurance Issues',
                                                                                         'Medical_consult': 'Consultation Issues',
                                                                                         'support': 'Insurance Issues',
                                                                                         'PriceBooking': 'Booking issue'
                                                                                         })
            
            N = negative_reason.value_counts(sort=False).drop(index='Location-Related')
            N = pd.DataFrame({'Negative reason': N.index, 'Count':  N.values})
            A = sns.barplot(y='Negative reason', x='Count', data=N, ax=ax_bar[0])
            A.axes.set_title("Negative feedback Complaints Categories",fontsize=15)
            A.set_xlabel('Count',fontsize=13)
            A.set_ylabel('Negative reason',fontsize=13)
            A.tick_params(labelsize=11)
            for index, row in N.iterrows():
                A.text(row.Count, row.name, row.Count, color='black', ha="left")
            
            # ============================================================================================================= #
            OBJ = self.data.query("predicted=='obj'")
            OBJ.loc[OBJ.inquiry==0, 'tags'] = 'gen-'
            objective_reason = OBJ.tags.str.rstrip('-').str.split('-').explode().replace({'': 'General Inquiry',
                                                                                          'price': 'Price Inquiry',
                                                                                          'service': 'Service Inquiry',
                                                                                          'datetime': 'Appointment Inquiry',
                                                                                          'place': 'Location Inquiry',
                                                                                          'structure': 'General Inquiry',
                                                                                          'insurance': 'Insurance Inquiry',
                                                                                          'support': 'Insurance Inquiry',
                                                                                          'Medical_consult': 'Medical-Consultation',
                                                                                          'PriceBooking': 'Booking-Inquiry',
                                                                                          'gen': 'General Conversation'
                                                                                         })
            
            O = objective_reason.value_counts(sort=False)
            O = pd.DataFrame({'Objective Categories': O.index, 'Count':  O.values})
            B = sns.barplot(y='Objective Categories', x='Count', data=O, ax=ax_bar[-1])
            B.axes.set_title("Objective Comments Categories",fontsize=15)
            B.set_xlabel('Count',fontsize=13)
            B.set_ylabel('Objective Categories',fontsize=13)
            B.tick_params(labelsize=11)
            for index, row in O.iterrows():
                B.text(row.Count, row.name, row.Count, color='black', ha="left")
        
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv("./data/data.csv")
    w2v_model = Word2Vec.load('./models/w2v_checkpoints/w2v_model_epoch15.gz') # can be skipped
    wtv_vect = WordVecVectorizer(w2v_model) # Class defined in utilities.py # can be skipped
    
    eval = Evaluate(df)
    transformed = eval.transform(vect_type="w2v", vect_model=wtv_vect)
    predictiones = eval.predict(input=transformed, model_name='FCNN_w2v_model')
    eval.add_dicts()
    eval.visualize()
    Final_data = eval.get_data()
    print(Final_data.sample(5, random_state=10))