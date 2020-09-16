# -*- coding: utf-8 -*-
from keras.preprocessing.sequence import pad_sequences
from config import data_len
import pickle
from mongodb import *
from email_header_help import *


def get_email(name="Kay Mann"):
    """
    :param name:
    :return:e
    """
    email_result_ = get_person_by_name3(name, content=1, Cc=1, To=1, Subject=1, Date=1)

    return list(email_result_)


def get_token():
    """
    :return:
    """
    print('get-token')
    my_token_word = pickle.load(open('50char-token.pkl', 'rb'))
    print('get-token ok')
    return my_token_word


token_content = get_token()


def build_data(name="Kate Symes"):
    """
    """
    emails = get_email(name)
    emails_v = []
    for email in emails:
        content_tmp = email['content']
        content_tmp = token_content.texts_to_sequences([content_tmp])
        content_tmp = pad_sequences(content_tmp, maxlen=data_len, padding='post')
        time_tmp = email['Date']
        time_tmp = time_tmp.split(' ')[4].split(':')[0]
        # subject_tmp = email['Subject']
        subject_tmp = getSubNum(email)
        # cc_tmp = email['Cc']
        cc_tmp = getCcNum(email)
        # to_tmp = email['To']
        to_tmp = getToNum(email)
        tmp = [content_tmp, int(time_tmp)]
        tmp.extend(cc_tmp)
        tmp.extend(to_tmp)
        tmp.extend(subject_tmp)
        emails_v.append(tmp)

    return emails_v

